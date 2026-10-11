"""Distributed fixed-step and patch-router inference for LingBot v2."""

from __future__ import annotations

import csv
import gc
import json
import math
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image

from methods.LingBotWorld.shared.adapter.lingbot_v2 import (
    decode_lingbot_video,
    encode_lingbot_condition,
    initialize_lingbot_frozen_models,
)
from methods.LingBotWorld.shared.adapter.scheduler import LingBotFastScheduler
from methods.LingBotWorld.shared.backbone import LingBotV2CausalAttentionBlock
from methods.LingBotWorld.shared.checkpoint import load_router_weights
from methods.core.runtime import load_checkpoint_manifest
from methods.core.inference import (
    InferenceArtifactWriter,
    InferenceSample,
    VideoArtifactWriter,
)
from methods.LingBotWorld.shared.data.dl3dv import (
    interpolate_camera_poses,
    relative_adjacent_poses,
)
from methods.LingBotWorld.shared.data.contracts import VAE_TEMPORAL_STRIDE
from methods.LingBotWorld.shared.distributed import (
    FSDPTopology,
    SingleSampleCoordinator,
    initialize_distributed_process_group,
    wrap_block_fsdp,
)
from methods.Wan21.shared.adapter.wan_cache import (
    make_cross_attention_cache,
    make_self_attention_cache,
)
from methods.LingBotWorld.shared.trainer.noise import initial_lingbot_noise
from methods.LingBotWorld.shared.trainer.rollout import LingBotRouterRollout


INFERENCE_MODES = {"fixed4", "router"}


def _inference_mode(args: Any) -> str:
    mode = str(getattr(args, "inference_mode", ""))
    if not mode:
        mode = "router" if str(args.mode) == "router" else "fixed4"
    if mode not in INFERENCE_MODES:
        raise ValueError(
            f"unsupported LingBot inference_mode {mode!r}; expected {sorted(INFERENCE_MODES)}"
        )
    if (mode == "router") != (str(args.mode) == "router"):
        raise ValueError(
            "router inference_mode requires mode=router; "
            "fixed4 requires mode=baseline"
        )
    return mode


class _InferenceRoot(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, payload: dict[str, Any]) -> Any:
        return self.generator(**payload)


class _CollectiveGeneratorProxy:
    """Expose RouterRollout metadata while routing forwards through FSDP."""

    def __init__(self, wrapped_root: torch.nn.Module, raw_generator: torch.nn.Module) -> None:
        self.wrapped_root = wrapped_root
        self.raw_generator = raw_generator
        self.model = raw_generator.model
        self.backbone_spec = raw_generator.backbone_spec

    def __call__(self, **kwargs: Any) -> Any:
        return self.wrapped_root(kwargs)

    def _convert_flow_pred_to_x0_video(self, **kwargs: Any) -> torch.Tensor:
        return self.raw_generator._convert_flow_pred_to_x0_video(**kwargs)

    def write_sparse_chunk_cache(self, **kwargs: Any) -> None:
        self.raw_generator.write_sparse_chunk_cache(**kwargs)


class LingBotChunkStrategy(Protocol):
    """Public extension point for experiment-specific per-chunk sampling."""

    name: str

    def run_chunk(
        self,
        *,
        rollout: LingBotRouterRollout,
        noisy_input: torch.Tensor,
        chunk_condition: dict[str, torch.Tensor],
        start: int,
        chunk_index: int,
        kv_cache: list[dict[str, torch.Tensor]],
        crossattn_cache: list[dict[str, torch.Tensor]],
        first_chunk_step: int,
        num_steps: int,
    ) -> Any: ...


class LingBotInputProvider(Protocol):
    """Public extension point for supplying inference inputs."""

    samples: list[dict[str, Any]]

    def prompt(self, sample: dict[str, Any]) -> str: ...

    def load_raw(
        self,
        sample: dict[str, Any],
        *,
        requested_pixel_frames: int,
        max_area: int,
        chunk_frames: int,
    ) -> dict[str, Any]: ...

    def provenance_fields(
        self, samples: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class _DenseFixed4Strategy:
    name = "fixed4"

    def run_chunk(self, **kwargs: Any) -> Any:
        return kwargs["rollout"].run_chunk_dense_fixed_steps(
            noisy_input=kwargs["noisy_input"],
            conditional_dict=kwargs["chunk_condition"],
            current_start_frame=kwargs["start"],
            kv_cache=kwargs["kv_cache"],
            crossattn_cache=kwargs["crossattn_cache"],
        )


class _RouterStrategy:
    name = "router"

    def run_chunk(self, **kwargs: Any) -> Any:
        rollout = kwargs["rollout"]
        noisy_input = kwargs["noisy_input"]
        if kwargs["chunk_index"] == 0:
            steps_map = torch.full(
                rollout.patch_shape(
                    noisy_input.shape[0],
                    noisy_input.shape[1],
                    noisy_input.shape[-2],
                    noisy_input.shape[-1],
                ),
                kwargs["first_chunk_step"],
                device=noisy_input.device,
                dtype=torch.long,
            )
            result = rollout.run_chunk_fixed_steps(
                noisy_input=noisy_input,
                conditional_dict=kwargs["chunk_condition"],
                current_start_frame=kwargs["start"],
                kv_cache=kwargs["kv_cache"],
                crossattn_cache=kwargs["crossattn_cache"],
                steps_map=steps_map,
            )
            result.chunk_log.update(
                {"forced": True, "selected_step": int(kwargs["first_chunk_step"])}
            )
            return result
        return rollout.run_router(
            noisy_input=noisy_input,
            conditional_dict=kwargs["chunk_condition"],
            current_start_frame=kwargs["start"],
            kv_cache=kwargs["kv_cache"],
            crossattn_cache=kwargs["crossattn_cache"],
            chunk_index=kwargs["chunk_index"],
            force_max_steps=False,
        )


def _enable_router_head(generator: torch.nn.Module, args: Any) -> None:
    generator.enable_patch_exit_router_prediction(
        hidden_dim=int(args.router_head_dim),
        initial_exit_bias=float(args.router_initial_exit_bias),
    )


def _prompt_for_sample(source: Path, sample: dict[str, Any]) -> str:
    prompt = dict(sample["prompt"])
    if prompt["kind"] == "inline":
        return str(prompt["text"])
    return (source / prompt["path"]).read_text(encoding="utf-8").strip()


def _official_preprocess_image(
    image: Image.Image,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Match upstream ToTensor -> normalize -> float bicubic preprocessing."""

    rgb = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
    frame = rgb.permute(2, 0, 1).contiguous().float().div_(255.0)
    frame.sub_(0.5).div_(0.5)
    return torch.nn.functional.interpolate(
        frame.unsqueeze(0),
        size=(int(height), int(width)),
        mode="bicubic",
    )[0]


def _load_raw_example(
    source: Path,
    sample: dict[str, Any],
    *,
    requested_pixel_frames: int,
    max_area: int,
    chunk_frames: int = 4,
) -> dict[str, Any]:
    sample_id = str(sample["id"])
    directory = source / sample.get("directory", f"examples/{sample_id}")
    poses = np.load(directory / "poses.npy").astype(np.float64)
    intrinsics = np.load(directory / "intrinsics.npy").astype(np.float64)
    chunk_frames = int(chunk_frames)
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    available = min(len(poses), len(intrinsics), int(requested_pixel_frames))
    available = (
        (available - 1) // VAE_TEMPORAL_STRIDE
    ) * VAE_TEMPORAL_STRIDE + 1
    latent_frames = (available - 1) // VAE_TEMPORAL_STRIDE + 1
    latent_frames -= latent_frames % chunk_frames
    if latent_frames < chunk_frames:
        raise ValueError(f"example {sample_id} has too few camera frames")
    pixel_frames = (latent_frames - 1) * VAE_TEMPORAL_STRIDE + 1
    poses = poses[:pixel_frames]
    source_times = np.arange(pixel_frames, dtype=np.float64)
    latent_times = np.linspace(0, pixel_frames - 1, latent_frames, dtype=np.float64)
    latent_poses = interpolate_camera_poses(poses, source_times, latent_times)
    relative_poses, translation_scale, static_camera = relative_adjacent_poses(latent_poses)
    with Image.open(directory / "image.jpg") as image:
        image = image.convert("RGB")
        height, width = _official_output_size(
            image.height, image.width, max_area=int(max_area)
        )
        first_frame = _official_preprocess_image(
            image,
            height=height,
            width=width,
        )
    scale = np.asarray(
        [width / 832.0, height / 480.0, width / 832.0, height / 480.0],
        dtype=np.float64,
    )
    latent_intrinsics = np.repeat(intrinsics[:1] * scale, latent_frames, axis=0).astype(np.float32)
    return {
        "prompt": [_prompt_for_sample(source, sample)],
        "first_frame": first_frame.unsqueeze(0),
        "camera_c2ws": torch.from_numpy(relative_poses).unsqueeze(0),
        "camera_intrinsics": torch.from_numpy(latent_intrinsics).unsqueeze(0),
        "sample_metadata": [
            {
                "sample_id": sample_id,
                "requested_pixel_frames": int(requested_pixel_frames),
                "actual_pixel_frames": pixel_frames,
                "latent_frames": latent_frames,
                "chunks": latent_frames // chunk_frames,
                "chunk_frames": chunk_frames,
                "height": height,
                "width": width,
                "translation_scale": float(translation_scale),
                "static_camera": bool(static_camera),
            }
        ],
    }


class OfficialLingBotInputProvider:
    """Read image, prompt and camera files directly from sample directories."""

    def __init__(self, source: str | Path) -> None:
        root = Path(source)
        examples = root / "examples" if (root / "examples").is_dir() else root
        self.source = root
        self.samples = [
            {
                "id": directory.name,
                "directory": directory.relative_to(root).as_posix(),
                "prompt": {
                    "kind": "file",
                    "path": (directory / "prompt.txt").relative_to(root).as_posix(),
                },
            }
            for directory in sorted(examples.iterdir()) if directory.is_dir()
        ]

    def prompt(self, sample: dict[str, Any]) -> str:
        return _prompt_for_sample(self.source, sample)

    def load_raw(
        self,
        sample: dict[str, Any],
        *,
        requested_pixel_frames: int,
        max_area: int,
        chunk_frames: int,
    ) -> dict[str, Any]:
        return _load_raw_example(
            self.source, sample,
            requested_pixel_frames=requested_pixel_frames,
            max_area=max_area,
            chunk_frames=chunk_frames,
        )

    def provenance_fields(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_root": str(self.source.resolve()),
            "sample_ids": [str(sample["id"]).zfill(2) for sample in samples],
            "sample_inputs": _selected_input_paths(self.source, samples),
        }


def _official_output_size(height: int, width: int, *, max_area: int) -> tuple[int, int]:
    """Reproduce upstream max-area sizing with VAE and patch alignment."""

    if min(height, width, max_area) <= 0:
        raise ValueError("image dimensions and max_area must be positive")
    aspect_ratio = height / float(width)
    latent_height = round(math.sqrt(max_area * aspect_ratio) // 8 // 2 * 2)
    latent_width = round(math.sqrt(max_area / aspect_ratio) // 8 // 2 * 2)
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("official max-area sizing produced an empty latent grid")
    return latent_height * 8, latent_width * 8


def _select_official_samples(
    args: Any,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requested = tuple(str(item).zfill(2) for item in args.inference_sample_ids)
    if requested:
        requested_set = set(requested)
        if len(requested_set) != len(requested):
            raise ValueError("inference_sample_ids must not contain duplicates")
        selected = [
            sample
            for sample in samples
            if str(sample["id"]).zfill(2) in requested_set
        ]
        missing = requested_set.difference(
            str(sample["id"]).zfill(2) for sample in selected
        )
        if missing:
            raise ValueError(f"unknown official sample ids: {sorted(missing)}")
        # Preserve the explicit request order instead of directory order.
        by_id = {str(sample["id"]).zfill(2): sample for sample in selected}
        return [by_id[sample_id] for sample_id in requested]
    maximum = int(args.inference_max_samples)
    if maximum <= 0:
        raise ValueError("inference_max_samples must be positive")
    selected = samples[:maximum]
    if not selected:
        raise ValueError("official example selection contains no samples")
    return selected


def _resolved_lingbot_timesteps(args: Any) -> list[int]:
    scheduler = LingBotFastScheduler(
        shift=float(args.scheduler_shift),
        num_train_timesteps=int(getattr(args, "num_train_timesteps", 1000)),
    )
    resolved = [
        int(item)
        for item in scheduler.fast_timesteps(
            indices=tuple(int(item) for item in args.timestep_indices)
        )
    ]
    return resolved


def _lingbot_router_runtime(args: Any) -> dict[str, Any]:
    resolved_timesteps = _resolved_lingbot_timesteps(args)
    return {
        "method": "LingBotWorld",
        "family": "lingbot_v2",
        "num_train_timesteps": int(args.num_train_timesteps),
        "timestep_indices": [int(item) for item in args.timestep_indices],
        "resolved_timesteps": resolved_timesteps,
        "max_steps": len(resolved_timesteps),
        "num_frame_per_block": int(args.num_frame_per_block),
        "first_chunk_step": int(args.first_chunk_step),
        "router_head_contract": "shared_binary_v1",
    }


def _runtime_contract_mismatches(
    checkpoint_contract: dict[str, Any] | None,
    runtime_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    if not checkpoint_contract:
        return [
            {
                "field": "router_runtime",
                "checkpoint": None,
                "runtime": runtime_contract,
                "reason": "checkpoint_metadata_missing",
            }
        ]
    return [
        {
            "field": key,
            "checkpoint": checkpoint_contract.get(key),
            "runtime": runtime_value,
            "reason": "value_mismatch",
        }
        for key, runtime_value in runtime_contract.items()
        if checkpoint_contract.get(key) != runtime_value
    ]


def _checkpoint_provenance(args: Any, *, router_mode: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "transformer_dir": str(args.transformer_dir),
        "transformer_index": args.transformer_index,
        "assets_dir": str(args.assets_dir),
        "architecture": OmegaConf.to_container(OmegaConf.create(args.architecture), resolve=True),
    }
    if router_mode:
        checkpoint = Path(str(args.checkpoint)).expanduser().resolve(strict=True)
        manifest_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        if not (manifest_dir / "manifest.json").is_file():
            value["router_checkpoint"] = {"path": str(checkpoint)}
            return value
        manifest = load_checkpoint_manifest(manifest_dir)
        runtime_contract = _lingbot_router_runtime(args)
        checkpoint_contract_value = manifest.metadata.get("router_runtime")
        checkpoint_contract = (
            dict(checkpoint_contract_value)
            if isinstance(checkpoint_contract_value, dict) and checkpoint_contract_value
            else None
        )
        value["router_checkpoint"] = {
            "path": str(checkpoint),
            "model_id": manifest.model_id,
            "stage": manifest.stage,
            "step": int(manifest.metadata.get("step", -1)),
            "checkpoint_contract": checkpoint_contract,
            "runtime_contract": runtime_contract,
            "contract_mismatches": _runtime_contract_mismatches(
                checkpoint_contract, runtime_contract
            ),
        }
    return value


def _selected_input_paths(source: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for sample in samples:
        sample_id = str(sample["id"]).zfill(2)
        directory = source / sample.get("directory", f"examples/{sample_id}")
        result[sample_id] = {
            name: str((directory / name).resolve())
            for name in ("image.jpg", "poses.npy", "intrinsics.npy", "prompt.txt")
        }
    return result


def _run_provenance(
    args: Any,
    *,
    input_provider: LingBotInputProvider,
    samples: list[dict[str, Any]],
    router_mode: bool,
) -> dict[str, Any]:
    inference_mode = _inference_mode(args)
    resolved_timesteps = _resolved_lingbot_timesteps(args)
    return {
        **input_provider.provenance_fields(samples),
        "model_id": str(args.model_id),
        "mode": inference_mode,
        "seed": int(args.seed),
        "requested_pixel_frames": int(args.inference_num_pixel_frames),
        "max_area": int(args.inference_max_area),
        "scheduler_shift": float(args.scheduler_shift),
        "num_train_timesteps": int(args.num_train_timesteps),
        "timestep_indices": [int(item) for item in args.timestep_indices],
        "resolved_timesteps": resolved_timesteps,
        "max_steps": len(resolved_timesteps),
        "chunk_size": int(args.num_frame_per_block),
        "chunk_latent_frames": int(args.num_frame_per_block),
        "first_chunk_step": int(args.first_chunk_step),
        "local_attn_size": int(args.local_attn_size),
        "sink_size": int(args.sink_size),
        "mixed_precision": bool(args.mixed_precision),
        "inference_cpu_offload": bool(args.inference_cpu_offload),
        "fsdp_enabled": bool(dict(args.fsdp).get("enabled", True)),
        "sequence_parallel_size": int(getattr(args, "ulysses_size", 1)),
        "router_exit_threshold": float(args.router_exit_threshold) if router_mode else None,
        "checkpoint": _checkpoint_provenance(args, router_mode=router_mode),
    }

def _aggregate_router_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate canonical N-stage NFE fields and retain four-step aliases."""

    total_router_nfe = sum(int(item["router_nfe"]) for item in summaries)
    total_dense_reference_nfe = sum(
        int(item["dense_reference_nfe"]) for item in summaries
    )
    if total_dense_reference_nfe <= 0:
        raise RuntimeError("dense reference NFE must be positive")
    max_steps_values = {int(item["max_steps"]) for item in summaries}
    if len(max_steps_values) != 1:
        raise RuntimeError(
            f"router chunks disagree on max_steps: {sorted(max_steps_values)}"
        )
    max_steps = max_steps_values.pop()
    patch_counts = {
        step: sum(
            int(
                dict(item.get("patch_counts") or {}).get(
                    str(step), dict(item.get("patch_counts") or {}).get(step, 0)
                )
            )
            for item in summaries
        )
        for step in range(1, max_steps + 1)
    }
    result: dict[str, Any] = {
        "router_nfe": total_router_nfe,
        "dense_reference_nfe": total_dense_reference_nfe,
        "max_steps": max_steps,
        "patch_counts": patch_counts,
        "cache_clear_forwards": sum(
            int(item.get("cache_clear_forwards", 0)) for item in summaries
        ),
        "cache_clear_tokens": sum(
            int(item.get("cache_clear_tokens", 0)) for item in summaries
        ),
        "nfe_ratio": total_router_nfe / float(total_dense_reference_nfe),
    }
    if max_steps == 4:
        result["full4_nfe"] = total_dense_reference_nfe
        result["total_full4_nfe"] = total_dense_reference_nfe
    return result


    finalize_error = coordinator.broadcast_object(finalize_error)
    if finalize_error is not None:
        raise RuntimeError(f"inference artifact finalization failed: {finalize_error}")
    coordinator.barrier()


def _tree_to_device(value: Any, device: torch.device) -> Any:
    """Move tensor leaves without changing condition-tree structure."""

    if torch.is_tensor(value):
        return value.detach().to(device=device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


def _latent_frame_count(metadata: dict[str, Any]) -> int:
    value = metadata.get("latent_frames", metadata.get("latent_frame_count"))
    if value is None:
        raise KeyError("LingBot sample metadata is missing its latent frame count")
    return int(value)


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise ValueError(f"module {type(module).__name__} has no parameters") from exc


def _assert_module_device(
    module: torch.nn.Module,
    *,
    expected_type: str,
    label: str,
) -> None:
    devices = {parameter.device.type for parameter in module.parameters()}
    devices.update(buffer.device.type for buffer in module.buffers())
    if devices != {expected_type}:
        raise RuntimeError(
            f"{label} residency mismatch: expected only {expected_type}, got {sorted(devices)}"
        )


def _cuda_memory_snapshot(device: torch.device, *, phase: str) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "phase": phase,
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        "free_mb": free_bytes / (1024**2),
        "total_mb": total_bytes / (1024**2),
    }


def _slice_condition(condition: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {
        "prompt_embeds": condition["prompt_embeds"],
        "i2v_condition": condition["i2v_condition"][:, start:end],
        "camera_rays": condition["camera_rays"][:, :, start:end],
    }


def _initialize_generator(args: Any, context: Any, *, router_mode: bool) -> tuple[Any, Any, Any]:
    from methods.LingBotWorld.shared.adapter.wrapper import LingBotDiffusionWrapper

    torch.manual_seed(int(args.seed))
    init_device: torch.device | str = torch.device("cpu") if context.is_rank0 else "meta"
    generator = LingBotDiffusionWrapper.from_pretrained(
        args.transformer_dir,
        transformer_index=args.transformer_index,
        architecture=dict(args.architecture),
        torch_dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
        device=init_device,
        local_attn_size=int(args.local_attn_size),
        sink_size=int(args.sink_size),
        load_weights=context.is_rank0,
        timestep_shift=float(args.scheduler_shift),
        num_train_timesteps=int(getattr(args, "num_train_timesteps", 1000)),
    )
    if router_mode:
        _enable_router_head(generator, args)
    root = _InferenceRoot(generator)
    if router_mode and context.is_rank0:
        root.load_state_dict(load_router_weights(args.checkpoint), strict=True)
    sequence_parallel_size = int(getattr(args, "ulysses_size", 1))
    if sequence_parallel_size < 1:
        raise ValueError("ulysses_size must be positive")
    if sequence_parallel_size > 1 and sequence_parallel_size != context.world_size:
        raise ValueError(
            "Ulysses sequence parallelism must span every rank: "
            f"ulysses_size={sequence_parallel_size} world_size={context.world_size}"
        )
    fsdp_config = dict(args.fsdp)
    fsdp_config["ulysses_enabled"] = sequence_parallel_size > 1
    if not bool(fsdp_config.get("enabled", True)):
        if context.world_size != 1:
            raise ValueError("unwrapped LingBot inference requires world_size=1")
        if sequence_parallel_size > 1:
            raise ValueError("Ulysses sequence parallelism requires FSDP-wrapped inference")
        root.to(device=context.device)
        root.eval()
        dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        topology = FSDPTopology(
            world_size=1,
            sharding_strategy="none",
            wrap_strategy="unwrapped_single_gpu_inference",
            block_class=(
                f"{LingBotV2CausalAttentionBlock.__module__}."
                f"{LingBotV2CausalAttentionBlock.__qualname__}"
            ),
            use_orig_params=True,
            cpu_offload=False,
            param_dtype=str(dtype),
            reduce_dtype=str(torch.float32),
            buffer_dtype=str(torch.float32),
            ignored_module_paths=(),
            ignored_modules_replicated=False,
        )
        return generator, root, topology
    # Sequence parallelism must be configured before wrapping, mirroring the
    # standalone repository's patch-then-wrap order.
    generator.model.configure_sequence_parallel(
        sequence_parallel_size,
        dist.group.WORLD if sequence_parallel_size > 1 else None,
    )
    wrapped, topology = wrap_block_fsdp(
        root,
        transformer_block_cls=LingBotV2CausalAttentionBlock,
        device=context.device,
        fsdp_config=fsdp_config,
    )
    wrapped.eval()
    return generator, wrapped, topology


@torch.no_grad()
def _sample_latents(
    *,
    generator: Any,
    wrapped: Any,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
    router_mode: bool,
    exit_threshold: float,
    noise_generator: torch.Generator,
    inference_mode: str | None = None,
    timestep_indices: tuple[int, ...] = (0, 250, 500, 750),
    chunk_frames: int = 4,
    first_chunk_step: int | None = None,
    chunk_strategy: LingBotChunkStrategy | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
    strategy = chunk_strategy or (_RouterStrategy() if router_mode else _DenseFixed4Strategy())
    mode = strategy.name
    if chunk_strategy is None and inference_mode not in {None, mode}:
        raise ValueError("inference mode and default chunk strategy disagree")
    proxy = _CollectiveGeneratorProxy(wrapped, generator)
    scheduler = generator.get_scheduler()
    timestep_indices = tuple(int(item) for item in timestep_indices)
    timesteps = scheduler.fast_timesteps(indices=timestep_indices).to(noise.device)
    num_steps = int(len(timesteps))
    if num_steps < 2:
        raise ValueError("LingBot inference requires at least two timesteps")
    chunk_frames = int(chunk_frames)
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if noise.shape[1] % chunk_frames:
        raise ValueError(
            "LingBot latent frames must contain complete chunks: "
            f"latent_frames={noise.shape[1]} chunk_frames={chunk_frames}"
        )
    first_chunk_step = num_steps if first_chunk_step is None else int(first_chunk_step)
    if not 1 <= first_chunk_step <= num_steps:
        raise ValueError(f"first_chunk_step must lie in [1, {num_steps}]")
    if mode == "fixed4" and num_steps != 4:
        raise ValueError("LingBot fixed4 mode requires exactly four timesteps")
    rollout = LingBotRouterRollout(
        generator=proxy,
        scheduler=scheduler,
        denoising_step_list=timesteps,
        frame_seq_length=(noise.shape[-2] // 2) * (noise.shape[-1] // 2),
        context_noise=0,
        exit_threshold=float(exit_threshold),
        noise_generator=noise_generator,
    )
    frame_seq_length = rollout.frame_seq_length
    spec = generator.backbone_spec
    cache_dtype = generator.load_dtype
    local_attn_frames = int(generator.model.local_attn_size)
    cache_capacity_frames = (
        int(noise.shape[1]) if local_attn_frames == -1 else local_attn_frames
    )
    # Ulysses keeps only this rank's heads in the self-attention cache; cross
    # attention is never sequence-parallel and keeps every head.
    sequence_parallel_size = int(getattr(generator.model, "sp_size", 1) or 1)
    kv_cache = make_self_attention_cache(
        spec,
        batch_size=1,
        capacity_tokens=cache_capacity_frames * frame_seq_length,
        dtype=cache_dtype,
        device=noise.device,
        num_heads=spec.num_heads // sequence_parallel_size,
    )
    crossattn_cache = make_cross_attention_cache(
        spec, batch_size=1, dtype=cache_dtype, device=noise.device
    )
    chunks: list[torch.Tensor] = []
    exit_maps: list[torch.Tensor] = []
    summaries: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, noise.shape[1], chunk_frames)):
        end = start + chunk_frames
        chunk_condition = _slice_condition(condition, start, end)
        noisy_input = noise[:, start:end]
        if noise.is_cuda:
            torch.cuda.synchronize(noise.device)
        chunk_started = time.perf_counter()
        result = strategy.run_chunk(
            rollout=rollout,
            noisy_input=noisy_input,
            chunk_condition=chunk_condition,
            start=start,
            chunk_index=chunk_index,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            first_chunk_step=first_chunk_step,
            num_steps=num_steps,
        )
        if noise.is_cuda:
            torch.cuda.synchronize(noise.device)
        chunk_time_sec = time.perf_counter() - chunk_started
        chunks.append(result.latents.detach())
        exit_maps.append(result.steps_map.detach().cpu())
        summaries.append(
            {
                "chunk_index": chunk_index,
                "start_frame": start,
                "num_frames": int(noisy_input.shape[1]),
                "chunk_latent_frames": int(chunk_frames),
                "resolved_timesteps": [int(item) for item in timesteps],
                "chunk_time_sec": float(chunk_time_sec),
                **result.chunk_log,
            }
        )
    return torch.cat(chunks, dim=1), exit_maps, summaries


def _video_to_uint8_thwc(video: torch.Tensor) -> torch.Tensor:
    frames = ((video[0] * 0.5 + 0.5).clamp(0, 1) * 255).round().byte()
    return frames.permute(0, 2, 3, 1).cpu()


def _run_single_gpu_phased_inference(
    args: Any,
    *,
    context: Any,
    coordinator: SingleSampleCoordinator,
    input_provider: LingBotInputProvider,
    samples: list[dict[str, Any]],
    provenance: dict[str, Any],
    destination: Path,
    router_mode: bool,
    artifact_writer: InferenceArtifactWriter,
    chunk_strategy: LingBotChunkStrategy | None,
) -> dict[str, Any]:
    """Run inference without co-resident frozen models and the 14B DiT.

    The GPU residency order is deliberately strict:

    1. T5 and VAE encode every selected condition while no DiT exists.
    2. Both frozen models move to CPU, then the DiT generates every latent.
    3. The DiT is destroyed, then the VAE returns to GPU for decoding.

    This path is inference-only and requires a one-rank, unwrapped runtime.
    """

    if context.world_size != 1 or not context.is_rank0:
        raise ValueError("phased LingBot inference requires one rank")
    if bool(dict(args.fsdp).get("enabled", True)):
        raise ValueError("phased LingBot inference requires FSDP to be disabled")
    inference_mode = chunk_strategy.name if chunk_strategy is not None else _inference_mode(args)

    expected_sample_ids = list(provenance["sample_ids"])
    artifact_writer.begin(destination, provenance)
    temporary = tempfile.TemporaryDirectory(prefix=".lingbot-phased-", dir=destination)
    workspace = Path(temporary.name)
    memory_phases: list[dict[str, Any]] = []

    # Phase 1: construct and use only the frozen condition encoders on CUDA.
    torch.cuda.reset_peak_memory_stats(context.device)
    try:
        text_encoder, vae = initialize_lingbot_frozen_models(
            args,
            device=context.device,
        )
    except Exception as exc:
        raise RuntimeError(
            f"single-GPU frozen-model initialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    _assert_module_device(text_encoder, expected_type="cuda", label="T5 condition encoder")
    _assert_module_device(vae, expected_type="cuda", label="VAE condition encoder")

    prepared: list[dict[str, Any]] = []
    for sample in samples:
        raw = input_provider.load_raw(
            sample,
            requested_pixel_frames=int(args.inference_num_pixel_frames),
            max_area=int(args.inference_max_area),
            chunk_frames=int(args.num_frame_per_block),
        )
        metadata = raw["sample_metadata"]
        condition = encode_lingbot_condition(
            raw,
            args=args,
            device=context.device,
            text_encoder=text_encoder,
            vae=vae,
        )
        prepared.append(
            {
                "sample": sample,
                "metadata": metadata,
                "condition": _tree_to_device(condition, torch.device("cpu")),
            }
        )
        del raw, condition
    torch.cuda.synchronize(context.device)
    memory_phases.append(_cuda_memory_snapshot(context.device, phase="condition_cuda"))

    text_encoder.to(device=torch.device("cpu"))
    vae.to(device=torch.device("cpu"))
    torch.cuda.synchronize(context.device)
    _assert_module_device(text_encoder, expected_type="cpu", label="offloaded T5")
    _assert_module_device(vae, expected_type="cpu", label="offloaded VAE")
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    memory_phases.append(
        _cuda_memory_snapshot(context.device, phase="condition_models_offloaded")
    )

    # Phase 2: no T5/VAE CUDA parameters may exist before constructing the DiT.
    _assert_module_device(vae, expected_type="cpu", label="pre-DiT VAE")
    torch.cuda.reset_peak_memory_stats(context.device)
    generator, wrapped, topology = _initialize_generator(
        args,
        context,
        router_mode=router_mode,
    )
    _assert_module_device(generator, expected_type="cuda", label="LingBot DiT")
    memory_phases.append(_cuda_memory_snapshot(context.device, phase="dit_loaded"))

    for sample_index, item in enumerate(prepared):
        sample = item["sample"]
        metadata = item["metadata"]
        condition = _tree_to_device(item["condition"], context.device)
        latent_frames = _latent_frame_count(metadata[0])
        latent_height, latent_width = map(int, condition["i2v_condition"].shape[-2:])
        noise, noise_generator = initial_lingbot_noise(
            (1, latent_frames, 16, latent_height, latent_width),
            device=context.device,
            compute_dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
            seed=int(args.seed),
        )
        rank0_noise = coordinator.broadcast_tree(noise)
        if not torch.equal(noise, rank0_noise):
            raise RuntimeError("LingBot initial CFHW noise diverged in one-rank inference")
        noise = rank0_noise
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        torch.cuda.synchronize(context.device)
        torch.cuda.reset_peak_memory_stats(context.device)
        started = time.perf_counter()
        latents, exit_maps, summaries = _sample_latents(
            generator=generator,
            wrapped=wrapped,
            noise=noise,
            condition=condition,
            router_mode=router_mode,
            exit_threshold=float(args.router_exit_threshold),
            noise_generator=noise_generator,
            inference_mode=inference_mode,
            timestep_indices=tuple(int(item) for item in args.timestep_indices),
            chunk_frames=int(args.num_frame_per_block),
            first_chunk_step=int(args.first_chunk_step),
            chunk_strategy=chunk_strategy,
        )
        torch.cuda.synchronize(context.device)
        inference_seconds = time.perf_counter() - started
        sample_id = str(sample["id"]).zfill(2)
        latent_path = workspace / f"{sample_id}.latents.pt"
        torch.save(latents.detach().cpu(), latent_path)
        torch.save(
            torch.stack(exit_maps),
            workspace / f"{sample_id}.exit-maps.pt",
        )
        timing = {
            **metadata[0],
            "sample_id": sample_id,
            "sample_index": sample_index,
            "prompt": input_provider.prompt(sample),
            "output_file": f"{sample_id}.mp4",
            "mode": inference_mode,
            "seed": int(args.seed),
            "inference_seconds": inference_seconds,
            "diffusion_gpu_time_sec": sum(
                float(summary["chunk_time_sec"]) for summary in summaries
            ),
            "chunk_times_sec": [
                float(summary["chunk_time_sec"]) for summary in summaries
            ],
            "generation_chunk_latent_frames": int(args.num_frame_per_block),
            "peak_allocated_mb": torch.cuda.max_memory_allocated(context.device) / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(context.device) / (1024**2),
        }
        torch.save(timing, workspace / f"{sample_id}.timing.pt")
        if router_mode:
            nfe_summary = _aggregate_router_summaries(summaries)
            summary_row = {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "prompt": input_provider.prompt(sample),
                "output_file": f"{sample_id}.mp4",
                "chunks": summaries,
                **nfe_summary,
                "actual_inference_time_sec": inference_seconds,
                "actual_chunk_times_sec": [
                    float(summary["chunk_time_sec"]) for summary in summaries
                ],
                "selected_steps": [
                    int(summary["selected_step"]) for summary in summaries
                ],
            }
            torch.save(summary_row, workspace / f"{sample_id}.router-summary.pt")
        del condition, noise, latents, exit_maps, summaries, noise_generator
        gc.collect()
        # Samples may have different aspect ratios. Release shape-specific
        # attention workspaces before allocating the next sample so the CUDA
        # caching allocator cannot accumulate incompatible 60x104 and 64x96
        # segments close to the 80-GiB device limit.
        torch.cuda.empty_cache()

    memory_phases.append(_cuda_memory_snapshot(context.device, phase="dit_rollout_complete"))
    dit_reference = weakref.ref(generator.model)
    del wrapped, generator
    gc.collect()
    torch.cuda.empty_cache()
    if dit_reference() is not None:
        raise RuntimeError("LingBot DiT is still referenced before VAE decode")
    released_snapshot = _cuda_memory_snapshot(context.device, phase="dit_released")
    memory_phases.append(released_snapshot)
    if (
        released_snapshot["allocated_mb"] > 1024.0
        or released_snapshot["reserved_mb"] > 2048.0
    ):
        raise RuntimeError(
            "LingBot DiT CUDA storage was not released before VAE decode: "
            f"allocated_mb={released_snapshot['allocated_mb']:.1f}, "
            f"reserved_mb={released_snapshot['reserved_mb']:.1f}"
        )

    # Phase 3: decode from the on-disk CPU latents after the DiT is gone.
    torch.cuda.reset_peak_memory_stats(context.device)
    vae.to(device=context.device, dtype=torch.float32)
    _assert_module_device(vae, expected_type="cuda", label="VAE decoder")
    generated = 0
    for sample_id in expected_sample_ids:
        latents = torch.load(
            workspace / f"{sample_id}.latents.pt",
            map_location=context.device,
            weights_only=True,
        )
        torch.cuda.synchronize(context.device)
        vae_started = time.perf_counter()
        video = decode_lingbot_video(latents, vae=vae)
        torch.cuda.synchronize(context.device)
        vae_gpu_time_sec = time.perf_counter() - vae_started
        timing_row = torch.load(
            workspace / f"{sample_id}.timing.pt",
            map_location="cpu",
            weights_only=False,
        )
        timing_row["vae_gpu_time_sec"] = float(vae_gpu_time_sec)
        timing_row["profile_total_time_sec"] = float(
            timing_row["diffusion_gpu_time_sec"] + vae_gpu_time_sec
        )
        tensors = {
            "latents": latents.detach().cpu(),
            "frames": video.detach().cpu(),
            "exit_maps": torch.load(
                workspace / f"{sample_id}.exit-maps.pt",
                map_location="cpu",
                weights_only=True,
            ),
        }
        router_summary_path = workspace / f"{sample_id}.router-summary.pt"
        if router_summary_path.is_file():
            timing_row["router_summary"] = torch.load(
                router_summary_path, map_location="cpu", weights_only=False
            )
        artifact_writer.write(
            destination,
            InferenceSample(
                sample_id=sample_id,
                video=_video_to_uint8_thwc(video),
                metadata=timing_row,
                tensors=tensors,
            ),
        )
        generated += 1
        del latents, video, tensors
        gc.collect()
    torch.cuda.synchronize(context.device)
    memory_phases.append(_cuda_memory_snapshot(context.device, phase="vae_decode"))
    vae.to(device=torch.device("cpu"))
    torch.cuda.synchronize(context.device)
    _assert_module_device(vae, expected_type="cpu", label="final offloaded VAE")
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    memory_phases.append(_cuda_memory_snapshot(context.device, phase="all_models_offloaded"))
    artifact_writer.finish(
        destination,
        {**provenance, "offload_strategy": "condition_then_dit_then_decode", "memory_phases": memory_phases},
    )
    temporary.cleanup()
    return {
        "status": "complete",
        "mode": inference_mode,
        "output_dir": str(destination),
        "generated": generated,
        "fsdp_topology": topology.as_dict(),
        "offload_strategy": "condition_then_dit_then_decode",
    }


def run_inference(
    args: Any,
    *,
    output_dir: str | Path | None = None,
    artifact_writer: InferenceArtifactWriter | None = None,
    chunk_strategy: LingBotChunkStrategy | None = None,
    input_provider: LingBotInputProvider | None = None,
) -> dict[str, Any]:
    context = initialize_distributed_process_group(
        backend=str(args.backend), required_world_size=int(args.required_world_size)
    )
    coordinator = SingleSampleCoordinator(context)
    inference_mode = chunk_strategy.name if chunk_strategy is not None else _inference_mode(args)
    router_mode = str(args.mode) == "router"
    if chunk_strategy is None and (inference_mode == "router") != router_mode:
        raise ValueError("inference mode and adapter mode disagree")
    provider = input_provider or OfficialLingBotInputProvider(args.validation_source)
    samples = _select_official_samples(args, provider.samples)
    provenance = _run_provenance(
        args,
        input_provider=provider,
        samples=samples,
        router_mode=router_mode,
    )
    provenance["mode"] = inference_mode
    expected_sample_ids = list(provenance["sample_ids"])
    destination = Path(str(output_dir or args.inference_output_dir))
    writer = artifact_writer or VideoArtifactWriter()
    if bool(args.inference_cpu_offload):
        return _run_single_gpu_phased_inference(
            args,
            context=context,
            coordinator=coordinator,
            input_provider=provider,
            samples=samples,
            provenance=provenance,
            destination=destination,
            router_mode=router_mode,
            artifact_writer=writer,
            chunk_strategy=chunk_strategy,
        )
    generator, wrapped, topology = _initialize_generator(args, context, router_mode=router_mode)
    text_encoder = None
    vae = None
    component_error = None
    if context.is_rank0:
        try:
            text_encoder, vae = initialize_lingbot_frozen_models(
                args,
                device=context.device,
                )
        except Exception as exc:
            component_error = f"{type(exc).__name__}: {exc}"
    component_error = coordinator.broadcast_object(component_error)
    if component_error is not None:
        raise RuntimeError(
            f"rank-zero frozen-model initialization failed: {component_error}"
        )
    if context.is_rank0:
        writer.begin(destination, provenance)
    coordinator.barrier()

    generated = 0
    for sample_index, sample in enumerate(samples):
        raw_error = None
        if context.is_rank0:
            try:
                raw = provider.load_raw(
                    sample,
                    requested_pixel_frames=int(args.inference_num_pixel_frames),
                    max_area=int(args.inference_max_area),
                    chunk_frames=int(args.num_frame_per_block),
                )
                metadata = raw["sample_metadata"]
            except Exception as exc:
                raw = None
                metadata = None
                raw_error = f"{type(exc).__name__}: {exc}"
        else:
            raw = None
            metadata = None
        raw_error = coordinator.broadcast_object(raw_error)
        if raw_error is not None:
            raise RuntimeError(f"official sample loading failed: {raw_error}")
        metadata = coordinator.broadcast_object(metadata)
        condition = coordinator.compute_on_rank0(
            lambda: encode_lingbot_condition(
                raw,  # type: ignore[arg-type]
                args=args,
                device=context.device,
                text_encoder=text_encoder,  # type: ignore[arg-type]
                vae=vae,  # type: ignore[arg-type]
            )
        )
        latent_frames = _latent_frame_count(metadata[0])
        latent_height, latent_width = map(int, condition["i2v_condition"].shape[-2:])
        noise, noise_generator = initial_lingbot_noise(
            (1, latent_frames, 16, latent_height, latent_width),
            device=context.device,
            compute_dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
            seed=int(args.seed),
        )
        rank0_noise = coordinator.broadcast_tree(noise if context.is_rank0 else None)
        if not torch.equal(noise, rank0_noise):
            raise RuntimeError("LingBot initial CFHW noise diverged across ranks")
        noise = rank0_noise
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        torch.cuda.synchronize(context.device)
        torch.cuda.reset_peak_memory_stats(context.device)
        started = time.perf_counter()
        latents, exit_maps, summaries = _sample_latents(
            generator=generator,
            wrapped=wrapped,
            noise=noise,
            condition=condition,
            router_mode=router_mode,
            exit_threshold=float(args.router_exit_threshold),
            noise_generator=noise_generator,
            inference_mode=inference_mode,
            timestep_indices=tuple(int(item) for item in args.timestep_indices),
            chunk_frames=int(args.num_frame_per_block),
            first_chunk_step=int(args.first_chunk_step),
            chunk_strategy=chunk_strategy,
        )
        torch.cuda.synchronize(context.device)
        inference_seconds = time.perf_counter() - started
        write_error = None
        if context.is_rank0:
            try:
                assert vae is not None
                torch.cuda.synchronize(context.device)
                vae_started = time.perf_counter()
                video = decode_lingbot_video(latents, vae=vae)
                torch.cuda.synchronize(context.device)
                vae_gpu_time_sec = time.perf_counter() - vae_started
                sample_id = str(sample["id"]).zfill(2)
                timing = {
                    **metadata[0],
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "prompt": provider.prompt(sample),
                    "output_file": f"{sample_id}.mp4",
                    "mode": inference_mode,
                    "seed": int(args.seed),
                    "inference_seconds": inference_seconds,
                    "diffusion_gpu_time_sec": sum(
                        float(item["chunk_time_sec"]) for item in summaries
                    ),
                    "chunk_times_sec": [
                        float(item["chunk_time_sec"]) for item in summaries
                    ],
                    "generation_chunk_latent_frames": int(args.num_frame_per_block),
                    "vae_gpu_time_sec": float(vae_gpu_time_sec),
                    "profile_total_time_sec": float(
                        sum(float(item["chunk_time_sec"]) for item in summaries)
                        + vae_gpu_time_sec
                    ),
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(context.device) / (1024**2),
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(context.device) / (1024**2),
                }
                if router_mode:
                    nfe_summary = _aggregate_router_summaries(summaries)
                    summary = {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "prompt": provider.prompt(sample),
                        "output_file": f"{sample_id}.mp4",
                        "chunks": summaries,
                        **nfe_summary,
                        "actual_inference_time_sec": inference_seconds,
                        "actual_chunk_times_sec": [
                            float(item["chunk_time_sec"]) for item in summaries
                        ],
                        "selected_steps": [
                            int(item["selected_step"]) for item in summaries
                        ],
                    }
                    timing["router_summary"] = summary
                writer.write(
                    destination,
                    InferenceSample(
                        sample_id=sample_id,
                        video=_video_to_uint8_thwc(video),
                        metadata=timing,
                        tensors={
                            "latents": latents.detach().cpu(),
                            "frames": video.detach().cpu(),
                            "exit_maps": torch.stack(exit_maps),
                        },
                    ),
                )
                generated += 1
            except Exception as exc:
                write_error = f"{type(exc).__name__}: {exc}"
        write_error = coordinator.broadcast_object(write_error)
        if write_error is not None:
            raise RuntimeError(f"rank-zero inference artifact write failed: {write_error}")
        coordinator.barrier()

    finalize_error = None
    if context.is_rank0:
        try:
            writer.finish(destination, provenance)
        except Exception as exc:
            finalize_error = f"{type(exc).__name__}: {exc}"
    finalize_error = coordinator.broadcast_object(finalize_error)
    if finalize_error is not None:
        raise RuntimeError(f"inference artifact finalization failed: {finalize_error}")
    coordinator.barrier()
    result = {
        "status": "complete" if context.is_rank0 else "worker_complete",
        "mode": inference_mode,
        "output_dir": str(destination),
        "generated": generated if context.is_rank0 else 0,
        "fsdp_topology": topology.as_dict(),
    }
    return result


__all__ = [
    "LingBotChunkStrategy",
    "LingBotInputProvider",
    "OfficialLingBotInputProvider",
    "run_inference",
]
