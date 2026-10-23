"""Collective helpers for H3 sequence-parallel inference."""
import torch.distributed as dist


def rank_zero_call(function):
    """Propagate rank-zero filesystem/decoder failures to all waiting SP peers."""
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = (function(), None)
        except Exception as exc:
            result[0] = (None, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(result, src=0)
    value, error = result[0]
    if error:
        raise RuntimeError(error)
    return value
