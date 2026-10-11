"""Rank-zero data coordination for an eight-way LingBot FSDP job.

Rank zero is the only rank that touches DL3DV, T5, the VAE, or LPIPS.  The
legacy single-sample path broadcasts one tensor tree, while router data
parallelism scatters one homogeneous tensor tree to each rank and gathers
per-rank tensors back to rank zero for centralized post-processing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.distributed as dist


_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def initialize_distributed_process_group(
    *,
    backend: str = "nccl",
    required_world_size: int = 8,
) -> DistributedContext:
    """Initialize torchrun and reject accidental non-eight-way training."""

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not dist.is_initialized():
        missing = [key for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if key not in os.environ]
        if missing:
            raise RuntimeError(
                "LingBot router training must be launched with torchrun; missing "
                + ", ".join(missing)
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL LingBot training requires CUDA")
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    if world_size != int(required_world_size):
        raise RuntimeError(
            "LingBot v2 14B router training requires the fixed eight-rank topology; "
            f"expected world_size={required_world_size}, got {world_size}"
        )
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def _tree_spec(value: Any, tensors: list[torch.Tensor]) -> Any:
    if torch.is_tensor(value):
        index = len(tensors)
        tensors.append(value)
        return {
            "__lingbot_tensor__": index,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _tree_spec(item, tensors) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__lingbot_tuple__": [_tree_spec(item, tensors) for item in value]}
    if isinstance(value, list):
        return [_tree_spec(item, tensors) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot broadcast sample leaf of type {type(value).__name__}")


def _restore_tree(spec: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(spec, dict) and "__lingbot_tensor__" in spec:
        return tensors[int(spec["__lingbot_tensor__"])]
    if isinstance(spec, dict) and "__lingbot_tuple__" in spec:
        return tuple(_restore_tree(item, tensors) for item in spec["__lingbot_tuple__"])
    if isinstance(spec, dict):
        return {key: _restore_tree(item, tensors) for key, item in spec.items()}
    if isinstance(spec, list):
        return [_restore_tree(item, tensors) for item in spec]
    return spec


def _tree_layout(spec: Any) -> Any:
    """Return a value-independent layout used to validate scatter inputs."""

    if isinstance(spec, dict) and "__lingbot_tensor__" in spec:
        return (
            "tensor",
            tuple(int(size) for size in spec["shape"]),
            str(spec["dtype"]),
        )
    if isinstance(spec, dict) and "__lingbot_tuple__" in spec:
        return ("tuple", tuple(_tree_layout(item) for item in spec["__lingbot_tuple__"]))
    if isinstance(spec, dict):
        return (
            "dict",
            tuple((key, _tree_layout(item)) for key, item in spec.items()),
        )
    if isinstance(spec, list):
        return ("list", tuple(_tree_layout(item) for item in spec))
    return ("value", type(spec).__qualname__)


def _tensor_specs(spec: Any) -> list[dict[str, Any]]:
    """Collect tensor leaf specifications in their stable tree index order."""

    result: list[dict[str, Any]] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict) and "__lingbot_tensor__" in node:
            result.append(node)
        elif isinstance(node, dict) and "__lingbot_tuple__" in node:
            for child in node["__lingbot_tuple__"]:
                collect(child)
        elif isinstance(node, dict):
            for child in node.values():
                collect(child)
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(spec)
    result.sort(key=lambda item: int(item["__lingbot_tensor__"]))
    return result


class SingleSampleCoordinator:
    """Coordinate rank-zero inputs and per-rank tensor products."""

    def __init__(self, context: DistributedContext, *, source_rank: int = 0) -> None:
        self.context = context
        self.source_rank = int(source_rank)

    def broadcast_object(self, value: Any) -> Any:
        payload = [value if self.context.rank == self.source_rank else None]
        kwargs = {}
        if self.context.backend == "nccl":
            kwargs["device"] = self.context.device
        dist.broadcast_object_list(payload, src=self.source_rank, **kwargs)
        return payload[0]

    def all_gather_objects(self, value: Any) -> list[Any]:
        """Collect one small Python value from every rank in rank order."""

        values: list[Any] = [None] * self.context.world_size
        dist.all_gather_object(values, value)
        return values

    def raise_if_any_rank_error(
        self,
        local_error: str | None,
        *,
        stage: str,
    ) -> None:
        """Raise the same error on every rank after a rank-local operation."""

        errors = self.all_gather_objects(local_error)
        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if failures:
            raise RuntimeError(
                f"LingBot {stage} failed on {len(failures)} rank(s): "
                + " | ".join(failures)
            )

    def broadcast_tree(self, value: Any | None) -> Any:
        source_tensors: list[torch.Tensor] = []
        source_spec = (
            _tree_spec(value, source_tensors)
            if self.context.rank == self.source_rank
            else None
        )
        spec = self.broadcast_object(source_spec)
        tensor_specs = _tensor_specs(spec)
        received: list[torch.Tensor] = []
        for item in tensor_specs:
            dtype_name = str(item["dtype"])
            if dtype_name not in _DTYPES:
                raise TypeError(f"unsupported broadcast dtype {dtype_name!r}")
            index = int(item["__lingbot_tensor__"])
            if self.context.rank == self.source_rank:
                tensor = source_tensors[index].detach().contiguous().to(self.context.device)
            else:
                tensor = torch.empty(
                    tuple(int(size) for size in item["shape"]),
                    dtype=_DTYPES[dtype_name],
                    device=self.context.device,
                )
            dist.broadcast(tensor, src=self.source_rank)
            received.append(tensor)
        return _restore_tree(spec, received)

    def scatter_trees(self, values_on_rank0: Sequence[Any] | None) -> Any:
        """Scatter one homogeneous tensor tree from rank zero to each rank.

        Tensor leaves must have the same shape and dtype at the corresponding
        location in every tree.  Non-tensor leaves may hold different values,
        but their container layout and Python leaf types must match.  Tensor
        payloads use :func:`torch.distributed.scatter`; the small per-rank tree
        specifications are broadcast once so metadata values are preserved.
        """

        source_trees: list[Any] | None = None
        source_specs: list[Any] | None = None
        source_tensors: list[list[torch.Tensor]] | None = None
        source_exception: Exception | None = None
        outcome: dict[str, Any] | None = None
        if self.context.rank == self.source_rank:
            try:
                if values_on_rank0 is None:
                    raise ValueError("source rank requires scatter tree values")
                source_trees = list(values_on_rank0)
                if len(source_trees) != self.context.world_size:
                    raise ValueError(
                        "scatter tree count must equal world_size: "
                        f"got {len(source_trees)}, expected {self.context.world_size}"
                    )
                source_specs = []
                source_tensors = []
                for value in source_trees:
                    tensors: list[torch.Tensor] = []
                    source_specs.append(_tree_spec(value, tensors))
                    source_tensors.append(tensors)
                expected_layout = _tree_layout(source_specs[0])
                for rank, spec in enumerate(source_specs[1:], start=1):
                    if _tree_layout(spec) != expected_layout:
                        raise ValueError(
                            "scatter tensor trees must be homogeneous; "
                            f"rank {rank} layout differs from rank 0"
                        )
                for item in _tensor_specs(source_specs[0]):
                    dtype_name = str(item["dtype"])
                    if dtype_name not in _DTYPES:
                        raise TypeError(f"unsupported scatter dtype {dtype_name!r}")
                outcome = {"ok": True}
            except Exception as exc:
                source_exception = exc
                outcome = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        outcome = self.broadcast_object(outcome)
        if not bool(outcome["ok"]):
            message = (
                "rank-zero tensor-tree scatter failed: "
                f"{outcome['error_type']}: {outcome['error']}"
            )
            if source_exception is not None:
                raise RuntimeError(message) from source_exception
            raise RuntimeError(message)

        specs = self.broadcast_object(source_specs)
        if not isinstance(specs, list) or len(specs) != self.context.world_size:
            raise RuntimeError("tensor-tree scatter received an invalid specification list")
        local_spec = specs[self.context.rank]
        local_tensor_specs = _tensor_specs(local_spec)
        received: list[torch.Tensor] = []
        for item in local_tensor_specs:
            dtype_name = str(item["dtype"])
            if dtype_name not in _DTYPES:
                raise TypeError(f"unsupported scatter dtype {dtype_name!r}")
            index = int(item["__lingbot_tensor__"])
            output = torch.empty(
                tuple(int(size) for size in item["shape"]),
                dtype=_DTYPES[dtype_name],
                device=self.context.device,
            )
            scatter_list = None
            if self.context.rank == self.source_rank:
                assert source_tensors is not None
                scatter_list = [
                    tensors[index].detach().contiguous().to(self.context.device)
                    for tensors in source_tensors
                ]
            dist.scatter(output, scatter_list=scatter_list, src=self.source_rank)
            received.append(output)
        return _restore_tree(local_spec, received)

    def gather_tensor(
        self,
        local_tensor: torch.Tensor,
        *,
        dst: int = 0,
    ) -> list[torch.Tensor] | None:
        """Gather one same-shaped tensor per rank to ``dst`` without autograd."""

        if not 0 <= int(dst) < self.context.world_size:
            raise ValueError(f"gather destination rank {dst} is outside the process group")
        local_spec = (
            {
                "ok": True,
                "shape": tuple(int(size) for size in local_tensor.shape),
                "dtype": str(local_tensor.dtype),
            }
            if torch.is_tensor(local_tensor)
            else {"ok": False, "type": type(local_tensor).__name__}
        )
        all_specs: list[Any] = [None] * self.context.world_size
        dist.all_gather_object(all_specs, local_spec)
        if not all(bool(spec.get("ok")) for spec in all_specs):
            raise TypeError(
                "gather_tensor requires a tensor on every rank; "
                f"got {all_specs}"
            )
        expected = (all_specs[0]["shape"], all_specs[0]["dtype"])
        if any((spec["shape"], spec["dtype"]) != expected for spec in all_specs[1:]):
            raise ValueError(
                "gather_tensor requires matching shapes and dtypes on every rank; "
                f"got {all_specs}"
            )
        assert torch.is_tensor(local_tensor)
        local = local_tensor.detach().contiguous().to(self.context.device)
        gathered = (
            [torch.empty_like(local) for _ in range(self.context.world_size)]
            if self.context.rank == int(dst)
            else None
        )
        dist.gather(local, gather_list=gathered, dst=int(dst))
        return gathered

    def scatter_scalars(
        self,
        values_on_rank0: Sequence[float | int | bool | torch.Tensor] | None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Scatter one rank-zero scalar to each rank and return a scalar tensor."""

        scalar_values: list[torch.Tensor] | None = None
        source_exception: Exception | None = None
        outcome: dict[str, Any] | None = None
        if self.context.rank == self.source_rank:
            try:
                if values_on_rank0 is None:
                    raise ValueError("source rank requires scalar scatter values")
                values = list(values_on_rank0)
                if len(values) != self.context.world_size:
                    raise ValueError(
                        "scalar count must equal world_size: "
                        f"got {len(values)}, expected {self.context.world_size}"
                    )
                scalar_values = []
                for rank, value in enumerate(values):
                    tensor = torch.as_tensor(value, dtype=dtype, device=self.context.device)
                    if tensor.numel() != 1:
                        raise ValueError(
                            f"scalar scatter value for rank {rank} has {tensor.numel()} elements"
                        )
                    scalar_values.append(tensor.detach().reshape(1).contiguous())
                outcome = {"ok": True}
            except Exception as exc:
                source_exception = exc
                outcome = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        outcome = self.broadcast_object(outcome)
        if not bool(outcome["ok"]):
            message = (
                "rank-zero scalar scatter failed: "
                f"{outcome['error_type']}: {outcome['error']}"
            )
            if source_exception is not None:
                raise RuntimeError(message) from source_exception
            raise RuntimeError(message)
        output = torch.empty((1,), dtype=dtype, device=self.context.device)
        dist.scatter(output, scatter_list=scalar_values, src=self.source_rank)
        return output[0]

    def next_rank0_sample(self, iterator: Iterable[Any] | None) -> Any:
        if self.context.is_rank0:
            if iterator is None:
                raise ValueError("rank zero requires a dataset iterator")
            sample = next(iterator)  # type: ignore[arg-type]
        else:
            sample = None
        return self.broadcast_tree(sample)

    def compute_on_rank0(self, function: Callable[[], Any]) -> Any:
        value = None
        source_exception: Exception | None = None
        outcome = None
        if self.context.is_rank0:
            try:
                value = function()
                outcome = {"ok": True}
            except Exception as exc:  # synchronize rank-zero-only failures
                source_exception = exc
                outcome = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        outcome = self.broadcast_object(outcome)
        if not bool(outcome["ok"]):
            message = (
                "rank-zero conditioning failed: "
                f"{outcome['error_type']}: {outcome['error']}"
            )
            if source_exception is not None:
                raise RuntimeError(message) from source_exception
            raise RuntimeError(message)
        return self.broadcast_tree(value)

    def noise(self, shape: tuple[int, ...], *, dtype: torch.dtype, seed: int) -> torch.Tensor:
        value = None
        if self.context.is_rank0:
            generator = torch.Generator(device=self.context.device)
            generator.manual_seed(int(seed))
            value = torch.randn(shape, generator=generator, dtype=dtype, device=self.context.device)
        result = self.broadcast_tree(value)
        if not torch.is_tensor(result):
            raise RuntimeError("noise broadcast did not return a tensor")
        return result

    def route_choice(
        self,
        *,
        num_chunks: int,
        chunk_min: int,
        chunk_max: int,
        step_choices: tuple[int, ...],
        seed: int,
    ) -> tuple[int, int]:
        choice = None
        if self.context.is_rank0:
            low = max(1, int(chunk_min))
            high = min(int(num_chunks) - 1, int(chunk_max))
            if low > high:
                raise ValueError(f"empty router chunk range [{low}, {high}]")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            chunk = int(torch.randint(low, high + 1, (1,), generator=generator).item())
            step_index = int(torch.randint(0, len(step_choices), (1,), generator=generator).item())
            choice = (chunk, int(step_choices[step_index]))
        result = self.broadcast_object(choice)
        return int(result[0]), int(result[1])

    def assert_finite(self, tensors: Iterable[torch.Tensor | None], *, stage: str) -> None:
        flag = torch.zeros((), dtype=torch.int32, device=self.context.device)
        if any(tensor is not None and not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            flag.fill_(1)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if bool(flag.item()):
            raise FloatingPointError(f"non-finite LingBot router value at {stage}")

    def barrier(self) -> None:
        dist.barrier()


# Backward-compatible name for older launch scripts; new code should not encode
# a particular pod name in the distributed topology API.
initialize_train3_process_group = initialize_distributed_process_group


__all__ = [
    "DistributedContext",
    "SingleSampleCoordinator",
    "initialize_distributed_process_group",
    "initialize_train3_process_group",
]
