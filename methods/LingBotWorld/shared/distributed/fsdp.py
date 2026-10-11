"""Block-level FSDP utilities for the LingBot v2 14B router model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass(frozen=True)
class FSDPTopology:
    world_size: int
    sharding_strategy: str
    wrap_strategy: str
    block_class: str
    use_orig_params: bool
    cpu_offload: bool
    param_dtype: str
    reduce_dtype: str
    buffer_dtype: str
    ulysses_enabled: bool = False
    ignored_module_paths: tuple[str, ...] = ()
    ignored_modules_replicated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    values = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return values[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported FSDP dtype {value!r}") from exc


def _resolve_ignored_modules(
    root: torch.nn.Module,
    ignored_modules: Iterable[torch.nn.Module] | None,
) -> tuple[tuple[torch.nn.Module, ...], tuple[str, ...]]:
    """Normalize ignored descendants and return their canonical root-relative paths."""

    if ignored_modules is None:
        return (), ()

    modules = tuple(ignored_modules)
    paths_by_identity: dict[int, str] = {}
    for path, candidate in root.named_modules():
        paths_by_identity.setdefault(id(candidate), path)

    paths: list[str] = []
    seen: set[int] = set()
    for candidate in modules:
        if not isinstance(candidate, torch.nn.Module):
            raise TypeError(
                "ignored_modules must contain torch.nn.Module instances, "
                f"got {type(candidate).__name__}"
            )
        identity = id(candidate)
        path = paths_by_identity.get(identity)
        if path is None:
            raise ValueError("every ignored module must be a descendant of the FSDP root")
        if path == "":
            raise ValueError("the FSDP root itself cannot be an ignored module")
        if identity not in seen:
            paths.append(path)
            seen.add(identity)
    return modules, tuple(paths)


def wrap_block_fsdp(
    module: torch.nn.Module,
    *,
    transformer_block_cls: type[torch.nn.Module],
    device: torch.device,
    fsdp_config: dict[str, Any],
    ignored_modules: Iterable[torch.nn.Module] | None = None,
) -> tuple[FSDP, FSDPTopology]:
    """Wrap the root and transformer blocks, leaving requested modules replicated.

    FSDP does not shard, materialize, or synchronize parameters below an ignored
    module. The caller therefore owns initialization and gradient synchronization
    for every replicated ignored module.
    """

    if not dist.is_initialized():
        raise RuntimeError("FSDP requires an initialized process group")
    # Ulysses shards activations inside attention only; the caller configures it
    # on the backbone before wrapping and records it here for provenance.
    ulysses_enabled = bool(fsdp_config.get("ulysses_enabled", False))
    strategy_name = str(fsdp_config.get("sharding_strategy", "full")).lower()
    strategies = {
        "full": ShardingStrategy.FULL_SHARD,
        "full_shard": ShardingStrategy.FULL_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if strategy_name not in strategies:
        raise ValueError(f"unsupported LingBot FSDP sharding strategy {strategy_name!r}")
    wrap_strategy = str(fsdp_config.get("wrap_strategy", "transformer_block"))
    if wrap_strategy != "transformer_block":
        raise ValueError("LingBot v2 14B requires fsdp.wrap_strategy=transformer_block")

    param_dtype = _dtype(fsdp_config.get("param_dtype", "bf16"))
    reduce_dtype = _dtype(fsdp_config.get("reduce_dtype", "fp32"))
    buffer_dtype = _dtype(fsdp_config.get("buffer_dtype", "fp32"))
    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        cast_forward_inputs=False,
        # Root-input casting recursively rebuilds dataclasses in PyTorch.
        # ChunkKVWorkspace is intentionally mutable, so rebuilding it would
        # collect KV pieces into a temporary copy while the rollout commits
        # the untouched original. The wrapper also owns the BF16 autocast
        # boundary and must retain the official FP32 latent trajectory.
        cast_root_forward_inputs=False,
    )
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_block_cls},
    )
    backward_prefetch = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }[str(fsdp_config.get("backward_prefetch", "backward_pre")).lower()]
    use_orig_params = bool(fsdp_config.get("use_orig_params", True))
    cpu_offload = bool(fsdp_config.get("cpu_offload", False))
    normalized_ignored_modules, ignored_module_paths = _resolve_ignored_modules(
        module,
        ignored_modules,
    )

    def materialize_meta(submodule: torch.nn.Module) -> None:
        direct_parameters = tuple(submodule.parameters(recurse=False))
        direct_buffers = tuple(submodule.buffers(recurse=False))
        if any(item.is_meta for item in (*direct_parameters, *direct_buffers)):
            submodule.to_empty(device=device, recurse=False)

    wrapped = FSDP(
        module,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=strategies[strategy_name],
        mixed_precision=mixed_precision,
        device_id=device,
        use_orig_params=use_orig_params,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        limit_all_gathers=bool(fsdp_config.get("limit_all_gathers", True)),
        backward_prefetch=backward_prefetch,
        sync_module_states=bool(fsdp_config.get("sync_module_states", False)),
        param_init_fn=materialize_meta,
        ignored_modules=normalized_ignored_modules or None,
    )
    topology = FSDPTopology(
        world_size=dist.get_world_size(),
        sharding_strategy=strategy_name,
        wrap_strategy=wrap_strategy,
        block_class=f"{transformer_block_cls.__module__}.{transformer_block_cls.__qualname__}",
        use_orig_params=use_orig_params,
        cpu_offload=cpu_offload,
        param_dtype=str(param_dtype),
        reduce_dtype=str(reduce_dtype),
        buffer_dtype=str(buffer_dtype),
        ulysses_enabled=ulysses_enabled,
        ignored_module_paths=ignored_module_paths,
        ignored_modules_replicated=bool(ignored_module_paths),
    )
    return wrapped, topology


def full_state_dict(module: FSDP) -> dict[str, Any]:
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, policy):
        return module.state_dict()


def full_optimizer_state_dict(module: FSDP, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        module,
        StateDictType.FULL_STATE_DICT,
        state_policy,
        optim_policy,
    ):
        return FSDP.optim_state_dict(module, optimizer)


def load_full_state_dict(module: FSDP, state_dict: dict[str, Any]) -> None:
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, policy):
        module.load_state_dict(state_dict, strict=True)


def load_full_optimizer_state_dict(
    module: FSDP,
    optimizer: torch.optim.Optimizer,
    full_state: dict[str, Any] | None,
) -> None:
    rank0_state = full_state if dist.get_rank() == 0 else None
    sharded = FSDP.optim_state_dict_to_load(module, optimizer, rank0_state)
    optimizer.load_state_dict(sharded)


def clip_grad_norm_(module: FSDP, max_norm: float) -> torch.Tensor:
    return module.clip_grad_norm_(float(max_norm))


__all__ = [
    "FSDPTopology",
    "clip_grad_norm_",
    "full_optimizer_state_dict",
    "full_state_dict",
    "load_full_optimizer_state_dict",
    "load_full_state_dict",
    "wrap_block_fsdp",
]
