from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_optim_state_dict(model, optimizer):
    """Gather a full optimizer state on rank 0 for strict resumption."""

    return FSDP.full_optim_state_dict(
        model,
        optimizer,
        rank0_only=True,
    )


def load_fsdp_optim_state_dict(model, optimizer, full_state_dict):
    """Scatter a rank-0 full optimizer state and load the local shard."""

    state = FSDP.scatter_full_optim_state_dict(
        full_state_dict if dist.get_rank() == 0 else None,
        model,
    )
    optimizer.load_state_dict(state)


def load_fsdp_state_dict(model, state_dict):
    """Load a full CPU state dict into an FSDP-wrapped module."""

    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
        model.load_state_dict(state_dict, strict=True)


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    cpu_offload=False,
    backward_prefetch="backward_pre",
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    backward_prefetch_policy = {
        "pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "post": BackwardPrefetch.BACKWARD_POST,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }
    if isinstance(backward_prefetch, BackwardPrefetch):
        resolved_backward_prefetch = backward_prefetch
    else:
        normalized_backward_prefetch = (
            "none" if backward_prefetch is None else str(backward_prefetch).strip().lower()
        )
        try:
            resolved_backward_prefetch = backward_prefetch_policy[
                normalized_backward_prefetch
            ]
        except KeyError as exc:
            choices = ", ".join(sorted(backward_prefetch_policy))
            raise ValueError(
                "backward_prefetch must be one of "
                f"{choices}, got {backward_prefetch!r}"
            ) from exc

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        backward_prefetch=resolved_backward_prefetch,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
    )
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        self.shadow = {}
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    @torch.no_grad()
    def load_full_state_dict(self, fsdp_module, state_dict):
        """Rebuild rank-local EMA shards from a full FSDP state dict.

        The online model is restored even when loading or rebuilding the EMA
        state fails.  A full checkpoint must not be assigned directly to the
        rank-local ``shadow`` mapping because their names and shapes differ.
        """

        live_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in fsdp_module.module.named_parameters()
        }
        live_buffers = {
            name: buffer.detach().cpu().clone()
            for name, buffer in fsdp_module.module.named_buffers()
        }
        try:
            load_fsdp_state_dict(fsdp_module, state_dict)
            self._init_shadow(fsdp_module)
        finally:
            for name, parameter in fsdp_module.module.named_parameters():
                if name in live_parameters:
                    parameter.copy_(
                        live_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
                    )
            for name, buffer in fsdp_module.module.named_buffers():
                if name in live_buffers:
                    buffer.copy_(live_buffers[name].to(device=buffer.device, dtype=buffer.dtype))

    def copy_to(self, fsdp_module):
        for n, p in fsdp_module.module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))

    @torch.no_grad()
    def full_state_dict(self, fsdp_module):
        live_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in fsdp_module.module.named_parameters()
        }
        try:
            for name, parameter in fsdp_module.module.named_parameters():
                if name not in self.shadow:
                    raise KeyError(f"EMA shadow is missing parameter {name!r}")
                parameter.copy_(
                    self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                )
            # FSDP already normalizes wrapper names in a full state dict.  The
            # gathered mapping is therefore the checkpoint payload directly.
            return fsdp_state_dict(fsdp_module)
        finally:
            for name, parameter in fsdp_module.module.named_parameters():
                if name in live_state:
                    parameter.copy_(
                        live_state[name].to(device=parameter.device, dtype=parameter.dtype)
                    )
