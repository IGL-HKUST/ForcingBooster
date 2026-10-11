import argparse
import importlib
import torch
import os
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json
from pathlib import Path

from methods.Wan21.shared.rollout import (
    AdaptiveCausalInferencePipeline,
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    RouterCausalInferencePipeline,
)
from methods.Wan21.shared.checkpoint.legacy import load_generator_checkpoint
from methods.Wan21.shared.data.dataset import TextDataset, TextImagePairDataset
from methods.Wan21.shared.utils.misc import set_seed
from methods.Wan21.shared.adapter.taew_wrapper import TAEW21VAEWrapper
from methods.Wan21.shared.adapter.wan_wrapper import WanVAEWrapper
from methods.core.runtime import load_checkpoint_manifest
from methods.core.inference import InferenceSample, VideoArtifactWriter

from methods.Wan21.shared.adapter.demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument(
    "--vae_checkpoint_path",
    type=str,
    default=None,
    help="Optional TAEW2.1 checkpoint. Providing it switches VAE encode/decode to TAEW2.1.",
)
parser.add_argument(
    "--vae_mode",
    choices=("sync", "async"),
    default="sync",
    help=(
        "VAE decode mode. 'sync' is the vanilla default; 'async' uses the "
        "selected backend's streaming decoder on a dedicated CUDA stream."
    ),
)
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--adaptive_seq_mode", choices=["none", "adaptive"], default=None, help="Adaptive inference mode override")
parser.add_argument("--adaptive_first_chunk_exit_step", type=int, default=None, help="Force all adaptive tokens in the first chunk to exit at this step count")
parser.add_argument("--profile_generation", action="store_true", help="Print generation profiling details")
parser.add_argument("--overwrite", action="store_true", help="Regenerate videos even if the output file already exists")
parser.add_argument(
    "--low_memory",
    action="store_true",
    help="Explicitly enable dynamic text-encoder swapping instead of relying on the free-memory heuristic",
)
parser.add_argument(
    "--artifact_writer",
    default=None,
    help="Optional module:attribute implementing the public inference writer protocol",
)
parser.add_argument("--report_timing", action="store_true",
                    help="Only tested on A800, for the Causal Forcing++ latency. Not make claims for other hardware like H100. For the result on H100, refer to the reported results in the Self Forcing paper.")
args = parser.parse_args()


def _load_artifact_writer(reference):
    if not reference:
        return VideoArtifactWriter(crf=None, quantization="truncate")
    module_name, separator, attribute = str(reference).partition(":")
    if not separator:
        raise ValueError("--artifact_writer must use module:attribute syntax")
    target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target()


artifact_writer = _load_artifact_writer(args.artifact_writer)

# Initialize distributed inference only for a real multi-process launch.  Some
# Kubernetes images export rank-like variables even for an ordinary one-process
# command; creating a world-size-one process group there adds warnings and can
# select the wrong dataset sampler.
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

free_vram_gb = get_cuda_free_memory_gb(device)
print(f'Free VRAM {free_vram_gb} GB')
low_memory = bool(args.low_memory) or free_vram_gb < 40

torch.set_grad_enabled(False)


def _load_checkpoint_into_pipeline(pipeline, checkpoint_path, use_ema, config):
    router_mode = (getattr(config, "router_seq_mode", "none") or "none") == "router"
    require_router_checkpoint = bool(
        getattr(config, "require_router_checkpoint", False)
    )
    allow_missing = (
        ("model.patch_exit_router_head.",)
        if router_mode and not require_router_checkpoint
        else ()
    )
    return load_generator_checkpoint(
        pipeline.generator,
        checkpoint_path,
        use_ema=use_ema,
        allow_missing_prefixes=allow_missing,
    )


def _router_runtime_provenance(config, pipeline, checkpoint_path):
    runtime_contract = {
        "method": "Wan21",
        "family": "wan",
        "num_train_timesteps": int(
            getattr(
                config,
                "num_train_timesteps",
                getattr(config, "num_train_timestep", 1000),
            )
        ),
        "denoising_steps": [float(item) for item in config.denoising_step_list],
        "warp_denoising_steps": bool(config.warp_denoising_step),
        "resolved_timesteps": [
            float(item) for item in pipeline.denoising_step_list
        ],
        "max_steps": int(len(pipeline.denoising_step_list)),
        "cache_clear_timestep": float(getattr(config, "context_noise", 0)),
        "cache_clear_counts_toward_nfe": False,
        "num_frame_per_block": int(config.num_frame_per_block),
        "first_chunk_step": int(pipeline.first_chunk_step),
        "router_head_contract": "shared_binary_v1",
    }
    checkpoint_contract = None
    manifest_path = None
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path)).expanduser().resolve(strict=False)
        directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        if (directory / "manifest.json").is_file():
            manifest_path = str((directory / "manifest.json").resolve())
            checkpoint_contract = load_checkpoint_manifest(directory).metadata.get(
                "router_runtime"
            )
    mismatches = (
        [
            {
                "field": "router_runtime",
                "checkpoint": None,
                "runtime": runtime_contract,
                "reason": "checkpoint_metadata_missing",
            }
        ]
        if not checkpoint_contract
        else [
            {
                "field": key,
                "checkpoint": checkpoint_contract.get(key),
                "runtime": value,
                "reason": "value_mismatch",
            }
            for key, value in runtime_contract.items()
            if checkpoint_contract.get(key) != value
        ]
    )
    return {
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_manifest": manifest_path,
        "checkpoint_contract": checkpoint_contract,
        "runtime_contract": runtime_contract,
        "contract_mismatches": mismatches,
    }

config = OmegaConf.load(args.config_path)
if args.adaptive_seq_mode is not None:
    config.adaptive_seq_mode = args.adaptive_seq_mode
if args.adaptive_first_chunk_exit_step is not None:
    config.adaptive_first_chunk_exit_step = args.adaptive_first_chunk_exit_step

# Initialize pipeline
adaptive_seq_mode = getattr(config, "adaptive_seq_mode", "none")
router_seq_mode = getattr(config, "router_seq_mode", "none") or "none"
model_kwargs = getattr(config, "model_kwargs", {})
model_name = getattr(model_kwargs, "model_name", None) or model_kwargs.get(
    "model_name", "Wan2.1-T2V-1.3B"
)
model_path = getattr(model_kwargs, "model_path", None) or model_kwargs.get("model_path")
vae = (
    TAEW21VAEWrapper(
        args.vae_checkpoint_path,
        streaming_decode=args.vae_mode == "async",
    )
    if args.vae_checkpoint_path
    else WanVAEWrapper(
        streaming_decode=args.vae_mode == "async",
        model_name=model_name,
        model_path=model_path,
    )
)
if router_seq_mode not in {"none", "router"}:
    raise ValueError(
        f"Unsupported router_seq_mode={router_seq_mode!r}; "
        "Use 'none' or 'router'."
    )
if router_seq_mode == "router":
    pipeline = RouterCausalInferencePipeline(config, device=device, vae=vae)
elif adaptive_seq_mode == "adaptive":
    pipeline = AdaptiveCausalInferencePipeline(config, device=device, vae=vae)
elif hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device, vae=vae)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device, vae=vae)

# Set the target precision before loading so the trained FP32 router weights
# are never rounded to BF16 along with the denoiser.
pipeline.generator.to(dtype=torch.bfloat16)
if router_seq_mode == "router":
    pipeline.generator.model.patch_exit_router_head.float()

checkpoint_path = args.checkpoint_path or getattr(config, "generator_ckpt", None)
if checkpoint_path:
    _load_checkpoint_into_pipeline(pipeline, checkpoint_path, args.use_ema, config)
router_runtime_provenance = (
    _router_runtime_provenance(config, pipeline, checkpoint_path)
    if router_seq_mode == "router"
    else None
)

print(
    f"[inference-config] {pipeline.generator.backbone_spec.summary()} "
    f"denoising_steps={len(getattr(pipeline, 'denoising_step_list', [])) or getattr(pipeline, 'sampling_steps', 0)} "
    f"denoising_timesteps={getattr(pipeline, 'denoising_step_list', torch.tensor([])).tolist()} "
    f"chunk_frames={getattr(config, 'num_frame_per_block', 1)}",
    flush=True,
)

# Keep the T5 encoder in its declared FP32 precision. The denoiser and the
# vanilla Wan VAE retain the BF16 inference precision used by this entrypoint.
if not args.vae_checkpoint_path:
    pipeline.vae.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
if args.vae_checkpoint_path:
    pipeline.vae.to(device=device, dtype=torch.float16)
else:
    pipeline.vae.to(device=device)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# The tracked default creates only the output directory and MP4 files. Local
# experiments can inject a writer that persists timing, latents, or decisions.
artifact_writer.begin(
    Path(args.output_folder),
    {"method": "Wan21", "router_runtime": router_runtime_provenance},
)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path) and not args.overwrite:
            print('Video has been generated. Pass!')
            continue
        sample_seed = int(args.seed) + int(idx)
        set_seed(sample_seed)
        print(f"[sample-seed] sample_index={i} dataset_index={idx} seed={sample_seed}", flush=True)
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path) and not args.overwrite:
            print('Video has been generated. Pass!')
            continue
        sample_seed = int(args.seed) + int(idx)
        set_seed(sample_seed)
        print(f"[sample-seed] sample_index={i} dataset_index={idx} seed={sample_seed}", flush=True)
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    sample_report_timing = args.report_timing and i >= 1
    inference_kwargs = dict(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
    )
    if adaptive_seq_mode == "adaptive":
        inference_kwargs.update(
            adaptive_seq_mode=adaptive_seq_mode,
            profile_generation=args.profile_generation or sample_report_timing,
        )
    else:
        inference_kwargs.update(
            report_timing=sample_report_timing,
            profile=args.profile_generation,
        )
    sample_wall_start = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.time()
    video, latents = pipeline.inference(**inference_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_sec = time.time() - inference_start
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    peak_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    if sample_report_timing:
        latency = pipeline.first_chunk_time
        elapsed = pipeline.last_generation_time
        num_pixel_frames = video.shape[1]
        fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
        print(f"[Sample {i}] {num_pixel_frames} frames, "
              f"latency ↓ {latency:.2f}s, FPS ↑ {fps:.2f}")
        # Only tested on A800, for the Causal Forcing++ paper latency & throughput.
        # Not make claims for other hardware like H100.
        # For the result on H100, refer to the reported results in the Self Forcing paper.
        # We do not guarantee that our FPS/latency measurement protocol is identical to that used in the Self Forcing paper.
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.clear_cache()

    output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
    write_start = time.time()
    sample_total_time_sec = time.time() - sample_wall_start
    chunk_times = getattr(pipeline, "last_chunk_times_sec", None)
    mode_label = (
        "router"
        if router_seq_mode == "router"
        else (
            "adaptive"
            if adaptive_seq_mode == "adaptive"
            else (
                f"diffusion_{getattr(pipeline, 'sampling_steps', 0)}step"
                if hasattr(pipeline, "sampling_steps")
                else f"dense_{len(getattr(pipeline, 'denoising_step_list', []))}step"
            )
        )
    )
    timing_record = {
        "sample_index": int(i),
        "dataset_index": int(idx),
        "sample_seed": int(sample_seed),
        "prompt": prompt,
        "output_file": os.path.basename(output_path),
        "mode": mode_label,
        "inference_time_sec": float(inference_time_sec),
        "write_video_time_sec": None,
        "sample_total_time_sec": float(sample_total_time_sec),
        "chunk_times_sec": [float(value) for value in chunk_times] if chunk_times else [],
        "generation_chunk_latent_frames": int(getattr(config, "num_frame_per_block", 1)),
        "vae_chunk_times_sec": [
            float(value) for value in getattr(pipeline, "last_vae_chunk_times_sec", [])
        ],
        "vae_chunk_latent_frames": [
            int(value) for value in getattr(pipeline, "last_vae_chunk_latent_frames", [])
        ],
        "vae_chunk_pixel_frames": [
            int(value) for value in getattr(pipeline, "last_vae_chunk_pixel_frames", [])
        ],
        "vae_chunk_start_offsets_sec": [
            float(value)
            for value in getattr(pipeline, "last_vae_chunk_start_offsets_sec", [])
        ],
        "vae_chunk_ready_offsets_sec": [
            float(value)
            for value in getattr(pipeline, "last_vae_chunk_ready_offsets_sec", [])
        ],
        "vae_backend": "taew2_1" if args.vae_checkpoint_path else "wan",
        "vae_mode": args.vae_mode,
        "diffusion_gpu_time_sec": getattr(pipeline, "last_profile_diffusion_time_sec", None),
        "vae_gpu_time_sec": getattr(pipeline, "last_profile_vae_time_sec", None),
        "vae_overlap_time_sec": getattr(pipeline, "last_profile_overlap_time_sec", None),
        "profile_total_time_sec": getattr(pipeline, "last_profile_total_time_sec", None),
        "peak_allocated_mb": float(peak_allocated_mb),
        "peak_reserved_mb": float(peak_reserved_mb),
    }
    print("[sample-timing] " + json.dumps(timing_record, ensure_ascii=True), flush=True)
    router_summary = getattr(pipeline, "last_router_summary", None)
    if router_summary is not None and local_rank == 0:
        summary_record = {
            "sample_index": int(i),
            "dataset_index": int(idx),
            "sample_seed": int(sample_seed),
            "prompt": prompt,
            "output_file": os.path.basename(output_path),
            "actual_inference_time_sec": float(inference_time_sec),
            "actual_sample_total_time_sec": float(sample_total_time_sec),
            "actual_chunk_times_sec": [float(value) for value in chunk_times] if chunk_times else [],
            "router_runtime": router_runtime_provenance,
            **router_summary,
        }
        timing_record["router_summary"] = summary_record
        print("[router-video-summary] " + json.dumps(summary_record, ensure_ascii=True), flush=True)
    artifact_writer.write(
        Path(args.output_folder),
        InferenceSample(
            sample_id=str(int(idx)).zfill(5),
            filename=os.path.basename(output_path),
            video=video[0],
            metadata=timing_record,
            tensors={
                "latents": latents[0].detach().cpu(),
                "exit_maps": torch.stack(
                    [item.detach().cpu() for item in getattr(pipeline, "last_exit_maps", [])]
                )
                if getattr(pipeline, "last_exit_maps", None)
                else torch.empty(0),
            },
        ),
    )
    timing_record["write_video_time_sec"] = time.time() - write_start

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
artifact_writer.finish(
    Path(args.output_folder),
    {"method": "Wan21", "router_runtime": router_runtime_provenance},
)
