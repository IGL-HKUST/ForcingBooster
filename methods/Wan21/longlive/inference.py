"""Fixed-step, sparse-router and prompt-switch generation."""
import json
import time

import torch
from omegaconf import OmegaConf

from methods.Wan21.shared.adapter.wan_wrapper import WanTextEncoder, WanVAEWrapper
from methods.Wan21.shared.adapter.wan_cache import configure_causal_attention_capacity, make_self_attention_cache, make_cross_attention_cache
from methods.Wan21.shared.adapter.denoising_schedule import resolve_wan_denoising_steps

from .cache import install_recache_attention, recache
from .config import resolve_path
from .rollout import LongLiveRollout
from .weights import build_generator, load_merged, apply_official_weights


def make_caches(generator, height, width, device):
    spec = generator.backbone_spec
    frame_tokens = spec.tokens_per_frame(height//8, width//8)
    configure_causal_attention_capacity(generator.model, 12*frame_tokens, frame_tokens)
    kv = make_self_attention_cache(spec, batch_size=1, capacity_tokens=12*frame_tokens,
                                   dtype=torch.bfloat16, device=device)
    text = make_cross_attention_cache(spec, batch_size=1, dtype=torch.bfloat16, device=device)
    for layer in kv:
        layer["_host_position"] = (0, 0)
    return kv, text, frame_tokens


@torch.no_grad()
def native_chunk(generator, rollout, latents, conditional, kv, text, start_frame):
    rollout.bind_noise(latents, start_frame)
    for index, timestep in enumerate(rollout.denoising_step_list):
        times = timestep.to(latents.device).expand(1, latents.shape[1])
        _, prediction = generator(noisy_image_or_video=latents, conditional_dict=conditional,
            timestep=times, kv_cache=kv, crossattn_cache=text,
            current_start=start_frame*rollout.frame_seq_length, write_cache=True)
        if index < rollout.num_steps-1:
            # Match the official scheduler's broadcast dtype; a scalar sigma
            # would round each intermediate multiplication to BF16 instead.
            next_time = rollout.denoising_step_list[index+1].to(latents.device).expand(latents.shape[0]*latents.shape[1])
            latents = rollout.scheduler.add_noise(prediction.flatten(0, 1),
                rollout._noise_bank[index].flatten(0, 1), next_time).unflatten(0, prediction.shape[:2])
    generator(noisy_image_or_video=prediction, conditional_dict=conditional,
              timestep=torch.zeros((1, prediction.shape[1]), device=prediction.device),
              kv_cache=kv, crossattn_cache=text, current_start=start_frame*rollout.frame_seq_length,
              write_cache=True)
    for layer in kv:
        layer["_host_position"] = (int(layer["global_end_index"]), int(layer["local_end_index"]))
        layer.pop("_router_workspace", None)
    return prediction


@torch.no_grad()
def infer(cfg):
    directory = resolve_path(cfg.output_dir)
    if (directory / "chunks.jsonl").exists() or (directory / "report.json").exists():
        raise FileExistsError(f"Use a fresh inference output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, directory / "config.yaml")
    torch.cuda.set_device(torch.device(cfg.device))
    if cfg.weight_mode == "official_lora":
        generator = build_generator(cfg, dtype=torch.float32)
        apply_official_weights(generator, cfg)
        generator.to(device=cfg.device, dtype=torch.bfloat16)
        metadata = {"model_id": "longlive-1.3b", "lora_merged": False,
                    "reference": "official base and LoRA separately cast to BF16"}
    else:
        generator = build_generator(cfg).to(device=cfg.device, dtype=torch.bfloat16)
        generator.model.enable_patch_exit_router_prediction(hidden_dim=int(cfg.head_dim), initial_exit_bias=float(cfg.initial_exit_bias))
        metadata = load_merged(generator, resolve_path(cfg.trained_path or cfg.merged_checkpoint), allow_router=True)
        generator.model.patch_exit_router_head.to(device=cfg.device, dtype=torch.float32)
    if cfg.mode == "router" and not metadata.get("router_trained", False) and cfg.forced_exit_step is None:
        raise ValueError("Router inference requires a trained checkpoint, or an explicit forced_exit_step test")
    generator.requires_grad_(False).eval()
    generator.model.predict_patch_exit_router = False
    install_recache_attention(generator)
    encoder = WanTextEncoder(model_path=cfg.assets_path).to(device=cfg.device, dtype=torch.bfloat16).eval()
    scheduler = generator.get_scheduler()
    steps = resolve_wan_denoising_steps(scheduler, cfg.denoising_steps, warp=True, device=cfg.device)
    kv, text, frame_tokens = make_caches(generator, cfg.height, cfg.width, cfg.device)
    rollout = LongLiveRollout(generator=generator, scheduler=scheduler, denoising_step_list=steps,
                             frame_seq_length=frame_tokens, exit_threshold=float(cfg.exit_threshold))
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    # All modes consume exactly the same initial and transition noise.
    noise = torch.randn((1, int(cfg.num_output_frames), 16, int(cfg.height)//8, int(cfg.width)//8),
                        device=cfg.device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    conditions = [encoder(text_prompts=[str(prompt)]) for prompt in cfg.prompts]
    torch.cuda.synchronize()
    prompt_seconds = time.perf_counter()-start
    encoder.cpu()
    del encoder
    torch.cuda.empty_cache()
    output, logs, maps = [], [], []
    common = result = None
    segment = 0
    generation_seconds = 0.0
    switches = [int(x) for x in cfg.switch_frame_indices]
    for start_frame in range(0, int(cfg.num_output_frames), 3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        switched = segment < len(switches) and start_frame == switches[segment]
        if switched:
            segment += 1
            history = torch.cat(output[-4:], dim=1)
            recache(generator, history, start_frame, conditions[segment], kv, text,
                    frame_tokens, global_sink=bool(cfg.global_sink))
        latents = noise[:, start_frame:start_frame+3]
        if cfg.mode == "native":
            prediction = native_chunk(generator, rollout, latents, conditions[segment], kv, text, start_frame)
            log = {"patch_counts": {4: 3*frame_tokens}, "patch_avg_step": 4.0}
        else:
            common = dict(noisy_input=latents, conditional_dict=conditions[segment],
                          current_start_frame=start_frame, kv_cache=kv, crossattn_cache=text)
            if start_frame == 0 or cfg.forced_exit_step is not None:
                step = 4 if start_frame == 0 else int(cfg.forced_exit_step)
                shape = rollout.patch_shape(1, 3, latents.shape[3], latents.shape[4])
                result = rollout.run_chunk_fixed_steps(**common, steps_map=torch.full(shape, step, device=cfg.device), write_cache=True)
            else:
                result = rollout.run_router(**common, chunk_index=start_frame//3, commit_cache=True)
            prediction, log = result.latents, result.chunk_log
            maps.append(result.steps_map.cpu())
            log.pop("speedup", None)  # NFE ratio is not a measured speedup.
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        generation_seconds += elapsed
        output.append(prediction.cpu())
        log.update(start_latent_frame=start_frame, seconds=elapsed, prompt_segment=segment,
                   recache=switched, cache_global_end=int(kv[0]["global_end_index"]),
                   cache_local_end=int(kv[0]["local_end_index"]))
        logs.append(log)
        with (directory / "chunks.jsonl").open("a") as stream:
            stream.write(json.dumps(log)+"\n")
        if start_frame % 30 == 0:
            print(f"Generated {start_frame+3}/{cfg.num_output_frames} latent frames", flush=True)
    latents_cpu = torch.cat(output, dim=1)
    if cfg.save_latents:
        torch.save(latents_cpu, directory / "latents.pt")
    if maps:
        torch.save(torch.cat(maps, dim=1), directory / "exit_steps.pt")
    generation_peak = torch.cuda.max_memory_allocated()
    # The last router-call kwargs otherwise keep the multi-GB KV cache alive.
    common = result = None
    del latents, prediction
    del generator, kv, text, conditions, rollout, noise
    torch.cuda.empty_cache()
    decode_seconds, pixel_frames = 0.0, 0
    if cfg.decode:
        import imageio.v2 as imageio
        vae = WanVAEWrapper(model_path=cfg.assets_path).to(device=cfg.device, dtype=torch.bfloat16).eval()
        vae.clear_cache()
        writer = imageio.get_writer(directory / "video.mp4", fps=int(cfg.fps), codec="libx264", quality=8, macro_block_size=None)
        try:
            for chunk in output:
                gpu_chunk = chunk.to(cfg.device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                pixels = vae.decode_to_pixel(gpu_chunk, use_cache=True)
                torch.cuda.synchronize()
                decode_seconds += time.perf_counter()-start
                frames = ((pixels[0].permute(0, 2, 3, 1).float()+1)*127.5).clamp(0, 255).byte().cpu().numpy()
                for frame in frames:
                    writer.append_data(frame)
                pixel_frames += len(frames)
            expected = int(cfg.num_output_frames)*4-3
            if pixel_frames != expected:
                raise AssertionError(f"Decoded {pixel_frames} frames, expected {expected}")
        finally:
            writer.close()
            vae.clear_cache()
    report = {"mode": str(cfg.mode), "weight_mode": str(cfg.weight_mode), "seed": int(cfg.seed), "latent_frames": int(cfg.num_output_frames),
              "pixel_frames": pixel_frames, "prompt_seconds": prompt_seconds,
              "generation_seconds": generation_seconds, "vae_seconds": decode_seconds,
              "total_compute_seconds": prompt_seconds+generation_seconds+decode_seconds,
              "peak_allocated_gib": max(generation_peak, torch.cuda.max_memory_allocated())/2**30,
              "timing_scope": "text encode + synchronized chunk generation including recache/its transfers + VAE; excludes loads/output transfers/encoding",
              "checkpoint_metadata": metadata, "output_dir": str(directory)}
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report
