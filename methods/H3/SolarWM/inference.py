"""Native, forced-exit and router inference with windowed video decoding."""
import gc
import json
import time
import torch
import torch.distributed as dist
from ..shared.config import path
from ..shared.dist import rank_zero_call
from ..shared.examples import read_manifest, sample_name, select_samples
from .router.cache import RawKVCache
from .router.model import RouterHead
from .router.rollout import RouterRollout
from .runtime import initialize, load_backbone, build_inputs, model_config


def infer(cfg):
    if cfg.mode == "router" and cfg.forced_exit_step is None and not cfg.trained_path:
        raise ValueError("Router inference requires weights.checkpoint or forced_exit_step")
    started = time.perf_counter()
    rank, device = initialize(cfg)
    from .backbone.layout import patchify_video, unpatchify_video
    from .backbone.full_length import full_length_geometry
    samples = select_samples(read_manifest(cfg), cfg)
    backbone = load_backbone(cfg)
    head = None
    if cfg.mode == "router" and cfg.forced_exit_step is None:
        head = RouterHead(hidden_dim=cfg.head_dim).to(device)
        bundle = torch.load(path(cfg.trained_path), map_location="cpu", weights_only=False)
        head.load_state_dict(bundle["head"], strict=True)
        head.eval().requires_grad_(False)
        del bundle
    rollout = RouterRollout(backbone, head, threshold=cfg.exit_threshold, kv_commit=cfg.kv_commit)
    load_seconds = time.perf_counter() - started
    output = path(cfg.output_dir)
    pending = []
    for sample in samples:
        inputs, noise, rng = build_inputs(cfg, sample, device, int(cfg.seed))
        cache, chunks, maps = RawKVCache(), [], []
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        for index in range(sample["rollout_latents"] // 5):
            n = noise[:, :, index * 5:(index + 1) * 5]
            transitions = rollout.transitions(n, rng, inputs)
            if cfg.mode == "native":
                result = rollout.native_chunk(inputs, n, index=index, cache=cache, transition_noise=transitions)
            else:
                rows = patchify_video(n)
                forced = 4 if index == 0 else cfg.forced_exit_step
                exits = torch.full((rows.shape[1],), int(forced), device=device, dtype=torch.long) if forced is not None else None
                result = rollout.chunk(inputs, rows, index=index, cache=cache,
                                       transition_noise=transitions, forced_steps=exits)
            if rank == 0:
                chunks.append(unpatchify_video(result.rows, 5, 48, 84).cpu())
                maps.append(result.steps.cpu().reshape(5, 24, 42))
                print(json.dumps(dict(sample=sample_name(sample), chunk=index)), flush=True)
            del result
        torch.cuda.synchronize()
        report = dict(sample=sample_name(sample), seed=int(cfg.seed), mode=cfg.mode,
                      forced_exit_step=cfg.forced_exit_step, load_seconds=load_seconds,
                      generation_seconds=time.perf_counter() - generation_started)
        def save_sample():
            target = output / sample_name(sample)
            target.mkdir(parents=True, exist_ok=True)
            steps = torch.cat(maps)
            if cfg.save_latents or cfg.decode:
                torch.save(torch.cat(chunks, dim=2), target / "latent.pt")
            if cfg.save_exit_steps:
                torch.save(steps, target / "exit_steps.pt")
            report["exit_counts"] = torch.bincount(steps.flatten(), minlength=5)[1:].tolist()
            report["mean_steps"] = float(steps.float().mean())
            pending.append((sample, target, report))
        rank_zero_call(save_sample)
        cache.clear()
        chunks.clear()
        maps.clear()
        del inputs, noise, rng, cache, n, transitions
    # Free the sharded generator before loading the decoder on rank zero.
    backbone.hook.remove()
    del rollout, backbone, head
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    def decode_pending():
        vae = None
        if cfg.decode:
            from .backbone.decode import decode_write
            if cfg.decoder == "taeh3":
                from .backbone.taehv import TAEHV
                vae = TAEHV(cfg.taeh3_path, arch_name="taeh3").to(device, torch.float16)
                vae.eval().requires_grad_(False)
            else:
                from .backbone.optional import load_conditioners
                vae = load_conditioners(model_config(cfg), device=device, qwen=False,
                                        audio_vae=False, schedulers=False).video_vae
                vae.set_attention_backend("native")
        for sample, target, report in pending:
            if vae is not None:
                decode_started = time.perf_counter()
                generated = torch.load(target / "latent.pt", map_location="cpu", weights_only=True)
                decode_write(vae, generated, full_length_geometry(sample["num_frames"]),
                             target / "video.mp4", 24, device, streaming=True)
                report["decode_seconds"] = time.perf_counter() - decode_started
                if not cfg.save_latents:
                    (target / "latent.pt").unlink()
            (target / "report.json").write_text(json.dumps(report, indent=2))
    rank_zero_call(decode_pending)
    dist.barrier()
    result = dict(samples=len(samples), output_dir=str(output)) if rank == 0 else None
    dist.destroy_process_group()
    return result
