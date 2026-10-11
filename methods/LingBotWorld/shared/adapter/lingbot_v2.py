"""Lifecycle adapter and conditioning components for LingBot-World v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from methods.core.errors import AdapterError
from methods.core.weights import directory, checkpoint, wan_assets, validate_shards, MODEL_PATTERNS


def _destroy_distributed_process_group() -> None:
    """Release NCCL/TCPStore state after a top-level lifecycle action."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_path(value: Any, *, follow_symlinks: bool = True) -> Path | None:
    if value in (None, "", False):
        return None
    path = Path(str(value)).expanduser()
    absolute = path if path.is_absolute() else (_repo_root() / path)
    return (
        absolute.resolve(strict=False)
        if follow_symlinks
        else absolute.absolute()
    )


def to_runtime_config(
    config: DictConfig,
    *,
    lifecycle: str = "train",
) -> DictConfig:
    """Flatten the public recipe without inheriting Wan stage defaults.

    Router training intentionally retains its fixed eight-rank contract.  The
    lifecycle selector only relaxes those *training* invariants for inference,
    where a one-rank phased-offload profile is useful and does not change
    optimizer semantics.
    """

    lifecycle = str(lifecycle).strip().lower()
    if lifecycle not in {"prepare", "train", "infer"}:
        raise AdapterError(f"unsupported LingBot lifecycle {lifecycle!r}")

    params = dict(config.params)
    if lifecycle == "train":
        from methods.Wan21.shared.router_objective import validate_router_training_config
        validate_router_training_config(params)
    runtime = dict(config.get("runtime") or {})
    artifacts = dict(config.artifacts)
    model = dict(config.get("model") or {})
    weights = dict(config.weights)
    configured_timestep_indices = tuple(
        int(item) for item in params.get("timestep_indices", (0, 250, 500, 750))
    )
    data = dict(config.data)
    fsdp = dict(runtime.get("fsdp") or {})
    output = _resolve_path(artifacts.get("output_dir"))
    source = _resolve_path(data.get("source"))
    resume = _resolve_path(artifacts.get("resume_checkpoint"))
    if lifecycle == "train" and weights.get("init") and resume is not None:
        raise AdapterError("weights.init and artifacts.resume_checkpoint are mutually exclusive")
    if output is None:
        raise AdapterError("LingBot recipe requires artifacts.output_dir")
    inference_checkpoint = None
    if lifecycle == "infer" and str(params.get("mode", "router")) == "router":
        if not weights.get("checkpoint"):
            raise AdapterError("Router inference requires weights.checkpoint")
        inference_checkpoint = checkpoint(weights["checkpoint"], filename="LingBot-World/model.pt")

    initialization = (checkpoint(weights["init"], filename="LingBot-World/model.pt")
                      if lifecycle == "train" and weights.get("init") else None)
    snapshot = directory(weights.get("model"), patterns=MODEL_PATTERNS +
                         tuple("transformers/" + p for p in MODEL_PATTERNS))
    transformer_dir = snapshot / "transformers" if (snapshot / "transformers").is_dir() else snapshot
    validate_shards(transformer_dir)
    assets_dir = wan_assets(weights.get("assets") or weights["model"])
    architecture = model.get("architecture")
    if architecture is None:
        # Existing experiment recipes may use the checkpoint's own config.
        architecture_path = transformer_dir / "config.json"
        if not architecture_path.is_file():
            architecture_path = snapshot / "config.json"
        architecture = {
            key: value
            for key, value in json.loads(
                architecture_path.read_text(encoding="utf-8")
            ).items()
            if not key.startswith("_")
        }

    mixed_precision = str(runtime.get("mixed_precision", "bf16")).lower() in {
        "bf16",
        "bfloat16",
        "true",
        "1",
    }
    betas = tuple(float(item) for item in params.get("betas", (0.0, 0.999)))
    if len(betas) != 2:
        raise AdapterError("params.betas must contain two values")
    resolved = {
        "model_id": str(config.get("model_id", transformer_dir.name)),
        "stage": str(config.get("stage", "router")),
        "mode": str(params.get("mode", "router")),
        "inference_mode": str(
            params.get(
                "inference_mode",
                "router" if str(params.get("mode", "router")) == "router" else "fixed4",
            )
        ),
        "snapshot_path": str(snapshot),
        "transformer_dir": str(transformer_dir),
        "transformer_index": None,
        "architecture": architecture,
        "assets_dir": str(assets_dir),
        "output_dir": str(output),
        "data_path": None if source is None else str(source),
        "dataset_split_interval": int(data.get("split_interval", 20)),
        "dataset_required_source_frames": int(data.get("required_source_frames", 115)),
        "dataset_max_scenes": data.get("max_scenes"),
        "dataset_max_samples": data.get("max_samples"),
        "resume_checkpoint": None if resume is None else str(resume),
        "initialization_checkpoint": None if initialization is None else str(initialization),
        "validation_source": str(
            _resolve_path(data.get("validation_source")) or source or ""
        ),
        "validation_output_dir": str(
            _resolve_path(artifacts.get("validation_output_dir")) or (output / "validation")
        ),
        "inference_output_dir": str(
            _resolve_path(artifacts.get("inference_output_dir")) or output
        ),
        "checkpoint": None if inference_checkpoint is None else str(inference_checkpoint),
        "backend": str(runtime.get("backend", "nccl")),
        "required_world_size": int(runtime.get("required_world_size", 8)),
        "mixed_precision": mixed_precision,
        "conditioning_mode": str(runtime.get("conditioning_mode", "per_rank_local")),
        "same_sample_across_ranks": bool(runtime.get("same_sample_across_ranks", False)),
        "no_ulysses": bool(runtime.get("no_ulysses", True)),
        # Inference-only Ulysses sequence parallelism; 1 keeps the FSDP-only path.
        "ulysses_size": int(runtime.get("ulysses_size", 1)),
        "fsdp": fsdp,
        "inference_cpu_offload": bool(runtime.get("inference_cpu_offload", False)),
        "seed": int(params.get("seed", runtime.get("seed", 42))),
        "num_frame_per_block": int(params.get("num_frame_per_block", 4)),
        "num_training_frames": int(params.get("num_training_frames", 20)),
        "num_pixel_frames": int(params.get("num_pixel_frames", 77)),
        "inference_num_pixel_frames": int(
            params.get("inference_num_pixel_frames", params.get("num_pixel_frames", 77))
        ),
        "inference_max_area": int(params.get("inference_max_area", 480 * 832)),
        "inference_max_samples": int(params.get("inference_max_samples", 6)),
        "inference_sample_ids": tuple(
            str(item).zfill(2) for item in params.get("inference_sample_ids", ())
        ),
        "image_size": tuple(int(item) for item in params.get("image_size", (480, 832))),
        "latent_size": tuple(int(item) for item in params.get("latent_size", (60, 104))),
        "scheduler_shift": float(params.get("scheduler_shift", 10.0)),
        "num_train_timesteps": int(params.get("num_train_timesteps", 1000)),
        "timestep_indices": configured_timestep_indices,
        "local_attn_size": int(params.get("local_attn_size", 18)),
        "sink_size": int(params.get("sink_size", 6)),
        "speculative_cache_frames": int(
            params.get("speculative_cache_frames", params.get("num_frame_per_block", 4))
        ),
        "first_chunk_step": int(
            params.get("first_chunk_step", len(configured_timestep_indices))
        ),
        "router_train_chunk_min": int(params.get("train_chunk_min", 1)),
        "router_train_chunk_max": int(params.get("train_chunk_max", 4)),
        "router_train_step_choices": tuple(
            int(item)
            for item in params.get(
                "train_step_choices", range(1, len(configured_timestep_indices))
            )
        ),
        "router_exit_threshold": float(params.get("exit_threshold", 0.5)),
        "router_initial_exit_bias": float(params.get("initial_exit_bias", -2.0)),
        "router_head_dim": int(params.get("head_dim", 2048)),
        "router_nfe_weight": float(params.get("nfe_weight", 0.03)),
        "learning_rate": float(params.get("learning_rate", 1e-4)),
        "beta1": betas[0],
        "beta2": betas[1],
        "weight_decay": float(params.get("weight_decay", 0.01)),
        "max_grad_norm": float(params.get("max_grad_norm", 1.0)),
        "batch_size": int(params.get("batch_size", 1)),
        "total_batch_size": int(params.get("total_batch_size", 8)),
        "max_train_steps": int(params.get("max_train_steps", 1000)),
        "progress_log_iters": int(params.get("progress_log_iters", 10)),
        "gc_interval": int(params.get("gc_interval", 50)),
        "final_checkpoint_only": bool(params.get("final_checkpoint_only", True)),
        "save_initial_checkpoint": bool(params.get("save_initial_checkpoint", False)),
        "save_ema": bool(params.get("save_ema", False)),
        "no_save": bool(runtime.get("no_save", False)),
        "resolved_config": OmegaConf.to_container(config, resolve=True),
    }
    if resolved["mode"] == "router" and lifecycle == "train":
        invariants = {
            "num_training_frames": (resolved["num_training_frames"], 20),
            "num_pixel_frames": (resolved["num_pixel_frames"], 77),
            "batch_size": (resolved["batch_size"], 1),
            "total_batch_size": (resolved["total_batch_size"], 8),
            "final_checkpoint_only": (resolved["final_checkpoint_only"], True),
            "save_initial_checkpoint": (resolved["save_initial_checkpoint"], False),
            "save_ema": (resolved["save_ema"], False),
            "required_world_size": (resolved["required_world_size"], 8),
            "no_ulysses": (resolved["no_ulysses"], True),
            "ulysses_size": (resolved["ulysses_size"], 1),
            "conditioning_mode": (resolved["conditioning_mode"], "per_rank_local"),
            "same_sample_across_ranks": (resolved["same_sample_across_ranks"], False),
        }
        bad = {key: pair for key, pair in invariants.items() if pair[0] != pair[1]}
        if bad:
            raise AdapterError(f"LingBot router invariant mismatch: {bad}")
        if int(resolved["max_train_steps"]) <= 0:
            raise AdapterError("LingBot router max_train_steps must be positive")

        block_frames = int(resolved["num_frame_per_block"])
        if block_frames <= 0:
            raise AdapterError("LingBot num_frame_per_block must be positive")
        timestep_indices = tuple(resolved["timestep_indices"])
        if len(timestep_indices) < 2:
            raise AdapterError("LingBot router requires at least two timestep_indices")
        if int(resolved["num_train_timesteps"]) <= 0:
            raise AdapterError("LingBot num_train_timesteps must be positive")
        if min(timestep_indices) < 0 or max(timestep_indices) >= int(
            resolved["num_train_timesteps"]
        ):
            raise AdapterError("LingBot timestep_indices exceed the scheduler grid")
        num_router_steps = len(timestep_indices)
        if not 1 <= int(resolved["first_chunk_step"]) <= num_router_steps:
            raise AdapterError(
                f"LingBot first_chunk_step must lie in [1, {num_router_steps}]"
            )
        train_step_choices = tuple(resolved["router_train_step_choices"])
        if not train_step_choices or any(
            not 1 <= step < num_router_steps for step in train_step_choices
        ):
            raise AdapterError(
                "LingBot train_step_choices must be a non-empty subset of "
                f"[1, {num_router_steps - 1}]"
            )
        training_frames = int(resolved["num_training_frames"])
        if training_frames % block_frames:
            raise AdapterError(
                "LingBot router num_training_frames must contain complete chunks: "
                f"num_training_frames={training_frames}, num_frame_per_block={block_frames}"
            )
        num_chunks = training_frames // block_frames
        train_chunk_min = int(resolved["router_train_chunk_min"])
        train_chunk_max = int(resolved["router_train_chunk_max"])
        if not 1 <= train_chunk_min <= train_chunk_max < num_chunks:
            raise AdapterError(
                "LingBot router training chunk range must select a non-prefix chunk: "
                f"range=[{train_chunk_min},{train_chunk_max}], num_chunks={num_chunks}"
            )

        # Counterfactual targets reserve a non-committing physical tail while the
        # model's semantic local-attention window remains unchanged.
        local_attn_frames = int(resolved["local_attn_size"])
        if local_attn_frames != -1:
            if block_frames > local_attn_frames:
                raise AdapterError(
                    "num_frame_per_block cannot exceed the local-attention cache window"
                )
            speculative_frames = int(resolved["speculative_cache_frames"])
            if speculative_frames < block_frames:
                raise AdapterError(
                    "speculative_cache_frames must hold one complete router chunk"
                )
            physical_cache_frames = local_attn_frames + speculative_frames
            max_non_rolling_chunk = (physical_cache_frames - block_frames) // block_frames
            if train_chunk_max > max_non_rolling_chunk:
                raise AdapterError(
                    "LingBot router speculative tail is too small for the selected chunk: "
                    f"train_chunk_max={train_chunk_max}, maximum_non_rolling_chunk="
                    f"{max_non_rolling_chunk}, physical_cache_frames={physical_cache_frames}, "
                    f"num_frame_per_block={block_frames}"
                )
        resolved["dataset_profile"] = "train"
    elif lifecycle == "infer":
        inference_mode = str(resolved["inference_mode"])
        if inference_mode not in {"fixed4", "router"}:
            raise AdapterError(
                "LingBot inference_mode must be either fixed4 or router"
            )
        if (inference_mode == "router") != (resolved["mode"] == "router"):
            raise AdapterError(
                "LingBot router inference_mode requires params.mode=router, while "
                "fixed4 requires params.mode=baseline"
            )
        timestep_indices = tuple(resolved["timestep_indices"])
        if len(timestep_indices) < 2:
            raise AdapterError("LingBot inference requires at least two timestep_indices")
        num_train_timesteps = int(resolved["num_train_timesteps"])
        if num_train_timesteps <= 0:
            raise AdapterError("LingBot num_train_timesteps must be positive")
        if min(timestep_indices) < 0 or max(timestep_indices) >= num_train_timesteps:
            raise AdapterError("LingBot timestep_indices exceed the scheduler grid")
        if inference_mode == "fixed4" and len(timestep_indices) != 4:
            raise AdapterError("LingBot fixed4 mode requires exactly four timestep_indices")
        block_frames = int(resolved["num_frame_per_block"])
        if block_frames <= 0:
            raise AdapterError("LingBot num_frame_per_block must be positive")
        if int(resolved["speculative_cache_frames"]) < block_frames:
            raise AdapterError(
                "speculative_cache_frames must hold one complete inference chunk"
            )
        local_attn_frames = int(resolved["local_attn_size"])
        if local_attn_frames != -1 and block_frames > local_attn_frames:
            raise AdapterError(
                "num_frame_per_block cannot exceed the local-attention cache window"
            )
        num_router_steps = len(timestep_indices)
        if not 1 <= int(resolved["first_chunk_step"]) <= num_router_steps:
            raise AdapterError(
                f"LingBot first_chunk_step must lie in [1, {num_router_steps}]"
            )
        if int(resolved["required_world_size"]) <= 0:
            raise AdapterError("LingBot inference required_world_size must be positive")
        sequence_parallel_size = int(resolved["ulysses_size"])
        if sequence_parallel_size < 1:
            raise AdapterError("LingBot ulysses_size must be positive")
        if sequence_parallel_size > 1:
            if sequence_parallel_size != int(resolved["required_world_size"]):
                raise AdapterError(
                    "LingBot Ulysses sequence parallelism must span every rank: "
                    f"ulysses_size={sequence_parallel_size}, "
                    f"required_world_size={resolved['required_world_size']}"
                )
            if not bool(fsdp.get("enabled", True)):
                raise AdapterError(
                    "LingBot Ulysses sequence parallelism requires runtime.fsdp.enabled=true"
                )
            if bool(resolved["inference_cpu_offload"]):
                raise AdapterError(
                    "LingBot Ulysses sequence parallelism is incompatible with phased CPU offload"
                )
        if bool(resolved["inference_cpu_offload"]):
            if int(resolved["required_world_size"]) != 1:
                raise AdapterError(
                    "LingBot inference_cpu_offload currently requires world_size=1"
                )
            if bool(fsdp.get("enabled", True)):
                raise AdapterError(
                    "LingBot one-GPU phased offload requires runtime.fsdp.enabled=false"
                )
    return OmegaConf.create(resolved)


def _normalize_lingbot_raw_batch(
    raw_batch: Mapping[str, Any],
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Normalize either an uncollated sample or a batch-size-one collate."""

    raw_prompts = raw_batch["prompt"]
    prompts = [raw_prompts] if isinstance(raw_prompts, str) else list(raw_prompts)
    if len(prompts) != 1 or not isinstance(prompts[0], str):
        raise ValueError("each LingBot raw batch must contain one prompt")

    first_frame = raw_batch["first_frame"]
    c2ws = raw_batch["camera_c2ws"]
    intrinsics = raw_batch["camera_intrinsics"]
    if not all(torch.is_tensor(value) for value in (first_frame, c2ws, intrinsics)):
        raise TypeError("LingBot image and camera raw-batch fields must be tensors")
    if first_frame.ndim == 3:
        first_frame = first_frame.unsqueeze(0)
    if c2ws.ndim == 3:
        c2ws = c2ws.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if first_frame.ndim != 4 or first_frame.shape[0] != 1:
        raise ValueError(
            "each LingBot first_frame batch must have shape [1,3,H,W], got "
            f"{tuple(first_frame.shape)}"
        )
    if c2ws.ndim != 4 or c2ws.shape[0] != 1:
        raise ValueError(
            "each LingBot camera_c2ws batch must have shape [1,F,4,4], got "
            f"{tuple(c2ws.shape)}"
        )
    if intrinsics.ndim != 3 or intrinsics.shape[0] != 1:
        raise ValueError(
            "each LingBot camera_intrinsics batch must have batch size one, got "
            f"{tuple(intrinsics.shape)}"
        )

    raw_metadata = raw_batch.get("sample_metadata") or {}
    if isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        metadata_items = list(raw_metadata)
        if len(metadata_items) != 1 or not isinstance(metadata_items[0], Mapping):
            raise ValueError(
                "each LingBot raw batch must contain exactly one metadata mapping"
            )
        metadata = dict(metadata_items[0])
    return prompts[0], first_frame, c2ws, intrinsics, metadata


def initialize_lingbot_frozen_models(
    args: Any,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load the frozen T5 and conditioning VAE once per rank."""

    if device.type != "cuda":
        raise RuntimeError("LingBot frozen models require CUDA")
    from methods.Wan21.shared.adapter.wan_wrapper import WanTextEncoder, WanVAEWrapper

    compute_dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    text_encoder = (
        WanTextEncoder(model_path=str(args.assets_dir))
        .eval()
        .requires_grad_(False)
        .to(device=device, dtype=compute_dtype)
    )
    vae = (
        WanVAEWrapper(model_path=str(args.assets_dir))
        .eval()
        .requires_grad_(False)
        .to(device=device, dtype=torch.float32)
    )
    return text_encoder, vae


@torch.no_grad()
def encode_lingbot_condition(
    raw_batch: Mapping[str, Any],
    *,
    args: Any,
    device: torch.device,
    text_encoder: torch.nn.Module,
    vae: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Encode one local sample using already-resident frozen models."""

    from methods.LingBotWorld.shared.backbone import (
        build_camera_origin_direction_rays,
        pack_camera_rays,
    )
    from methods.LingBotWorld.shared.data import (
        assemble_i2v_condition,
        make_i2v_encode_video,
    )

    compute_dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    prompt, first_frame, c2ws, intrinsics, metadata = _normalize_lingbot_raw_batch(
        raw_batch
    )
    prompt_embeds = text_encoder(text_prompts=[prompt])["prompt_embeds"].to(
        dtype=compute_dtype
    )
    first_frame = first_frame.to(device=device, dtype=torch.float32)
    c2ws = c2ws.to(device=device, dtype=torch.float32)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    height, width = map(int, first_frame.shape[-2:])
    video_frame_count = int(
        metadata.get(
            "actual_pixel_frames",
            metadata.get("video_frame_count", args.num_pixel_frames),
        )
    )
    encode_video = make_i2v_encode_video(first_frame, video_frame_count)
    encoded_ntchw = vae.encode_to_latent(encode_video)
    encoded_ncthw = encoded_ntchw.permute(0, 2, 1, 3, 4).contiguous()
    i2v_condition = assemble_i2v_condition(encoded_ncthw, video_frame_count)
    i2v_condition = (
        i2v_condition.permute(0, 2, 1, 3, 4)
        .contiguous()
        .to(dtype=torch.float32)
    )
    camera_rays = build_camera_origin_direction_rays(
        c2ws, intrinsics, height, width
    )
    camera_rays = pack_camera_rays(camera_rays, spatial_stride=(8, 8)).to(
        dtype=compute_dtype
    )
    return {
        "prompt_embeds": prompt_embeds,
        "i2v_condition": i2v_condition,
        "camera_rays": camera_rays,
    }


@torch.no_grad()
def decode_lingbot_video(
    latents: torch.Tensor,
    *,
    vae: torch.nn.Module,
) -> torch.Tensor:
    """Decode latents with an already-resident FP32 VAE."""

    return vae.decode_to_pixel(latents.to(dtype=torch.float32), use_cache=False)


class LingBotV2Adapter:
    """Expose baseline inference and patch-router training from YAML configs."""

    @staticmethod
    def _mode(config: DictConfig) -> str:
        return str(dict(config.params).get("mode", "router"))

    def prepare(self, config: DictConfig) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "prepared",
            "model_id": str(config.model_id),
            "mode": self._mode(config),
        }
        if str(config.data.get("format", "")) == "dl3dv_i2v_camera":
            from methods.LingBotWorld.shared.data import enumerate_dl3dv_scenes

            result["dataset"] = enumerate_dl3dv_scenes(
                _resolve_path(config.data.get("source")),
                split_interval=int(config.data.get("split_interval", 20)),
                required_source_frames=int(config.data.get("required_source_frames", 115)),
                max_scenes=config.data.get("max_scenes"),
            )["statistics"]
        else:
            source = _resolve_path(config.data.get("validation_source") or config.data.get("source"))
            if (source / "examples").is_dir():
                source = source / "examples"
            result["examples"] = len(
                [path for path in sorted(source.iterdir()) if (path / "image.jpg").is_file()]
            )
        return result

    def train(self, config: DictConfig) -> dict[str, Any]:
        if self._mode(config) == "baseline":
            raise AdapterError("baseline.yaml is inference-only and cannot train")
        args = to_runtime_config(config, lifecycle="train")
        from methods.LingBotWorld.shared.trainer import LingBotRouterTrainer

        try:
            return LingBotRouterTrainer(args).train()
        finally:
            _destroy_distributed_process_group()

    def infer(self, config: DictConfig) -> dict[str, Any]:
        args = to_runtime_config(config, lifecycle="infer")
        from .inference import run_inference

        try:
            return run_inference(args)
        finally:
            _destroy_distributed_process_group()

__all__ = [
    "LingBotV2Adapter",
    "decode_lingbot_video",
    "encode_lingbot_condition",
    "initialize_lingbot_frozen_models",
    "to_runtime_config",
]
