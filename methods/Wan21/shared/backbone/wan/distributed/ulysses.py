"""Ulysses sequence-parallel primitives for causal Wan inference.

Ulysses keeps attention mathematically identical to the single-GPU path: each
rank holds a slice of the sequence outside attention, and an ``all_to_all``
inside attention trades that sequence slice for a head slice so every rank sees
the full sequence for its own heads.  A second ``all_to_all`` restores the
sequence sharding before the output projection.

The helpers below are deliberately stateless -- ranks/groups are passed in by
the modules that use them (``sp_size``/``sp_group`` attributes on the attention
modules), so nothing here has to be configured globally.
"""

import torch
import torch.distributed as dist


def sequence_parallel_world_size(group=None) -> int:
    """Return the sequence-parallel group size, or 1 when not distributed."""

    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def sequence_parallel_rank(group=None) -> int:
    """Return this process' rank inside the sequence-parallel group."""

    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def all_to_all(x: torch.Tensor, scatter_dim: int, gather_dim: int, group=None) -> torch.Tensor:
    """Scatter along ``scatter_dim`` and gather along ``gather_dim``.

    ``x.shape[scatter_dim]`` must be divisible by the group size; callers pad
    the sequence dimension beforehand (see :func:`pad_to_multiple`).
    """

    world_size = sequence_parallel_world_size(group)
    if world_size == 1:
        return x
    if x.shape[scatter_dim] % world_size:
        raise ValueError(
            "all_to_all requires a divisible scatter dimension: "
            f"shape={tuple(x.shape)} scatter_dim={scatter_dim} world_size={world_size}"
        )
    inputs = [chunk.contiguous() for chunk in x.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(chunk) for chunk in inputs]
    dist.all_to_all(outputs, inputs, group=group)
    return torch.cat(outputs, dim=gather_dim).contiguous()


def gather_forward(x: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    """All-gather equally sized shards and concatenate them along ``dim``."""

    world_size = sequence_parallel_world_size(group)
    if world_size == 1:
        return x
    source = x.contiguous()
    outputs = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(outputs, source, group=group)
    return torch.cat(outputs, dim=dim).contiguous()


def padded_length(length: int, multiple: int) -> int:
    """Round ``length`` up to the next multiple of ``multiple``."""

    remainder = int(length) % int(multiple)
    return int(length) if remainder == 0 else int(length) + int(multiple) - remainder


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1) -> torch.Tensor:
    """Zero-pad ``x`` along ``dim`` so its length is a multiple of ``multiple``."""

    target = padded_length(x.shape[dim], multiple)
    if target == x.shape[dim]:
        return x
    shape = list(x.shape)
    shape[dim] = target - x.shape[dim]
    return torch.cat([x, x.new_zeros(shape)], dim=dim)


def shard_sequence(x: torch.Tensor, sp_size: int, rank: int, dim: int = 1) -> torch.Tensor:
    """Pad along ``dim`` to a multiple of ``sp_size`` and keep this rank's shard."""

    if sp_size == 1:
        return x
    padded = pad_to_multiple(x, sp_size, dim=dim)
    shard = padded.shape[dim] // sp_size
    return padded.narrow(dim, rank * shard, shard).contiguous()
