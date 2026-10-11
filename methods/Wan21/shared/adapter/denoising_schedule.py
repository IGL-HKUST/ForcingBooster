"""Wan-specific denoising schedule resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_wan_denoising_steps(
    scheduler,
    steps: Sequence[int | float] | torch.Tensor,
    *,
    warp: bool,
    device: torch.device | str | int | None = None,
    minimum_steps: int = 1,
) -> torch.Tensor:
    """Resolve the formal Wan denoising stages.

    Warped Wan recipes describe positions on the unshifted training grid.  The
    value ``num_train_timesteps`` selects the first scheduler entry.  Wan uses
    timestep zero only for the final clean-KV/cache-only forward, not for a
    denoising stage.  A single trailing zero is therefore accepted as a legacy
    cache-clear marker and removed before resolving the formal schedule.  Zero
    anywhere else is rejected.  Non-zero values retain their configured order
    and duplicates.
    """

    values = (
        steps.detach().cpu().to(dtype=torch.float64)
        if torch.is_tensor(steps)
        else torch.as_tensor(list(steps), dtype=torch.float64)
    )
    if values.ndim != 1:
        raise ValueError("denoising steps must be one-dimensional")
    zero_indices = torch.nonzero(values == 0, as_tuple=False).flatten()
    if zero_indices.numel():
        if zero_indices.numel() != 1 or int(zero_indices[0]) != values.numel() - 1:
            raise ValueError(
                "Wan timestep 0 is reserved for the trailing clean-KV/cache-only "
                "forward and cannot appear among formal denoising steps"
            )
        values = values[:-1]
    if values.numel() < int(minimum_steps):
        raise ValueError(
            f"Wan requires at least {minimum_steps} formal denoising steps "
            "after removing the optional trailing cache-clear marker"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("denoising steps must contain only finite values")

    if warp:
        rounded = values.round()
        if not torch.equal(values, rounded):
            raise ValueError("warped Wan denoising steps must be integer grid values")
        # Older scheduler facades and lightweight test doubles may expose only
        # the resolved grid.  Treat its length as the legacy training-grid
        # contract while all maintained wrappers provide the explicit field.
        num_train_timesteps = int(
            getattr(scheduler, "num_train_timesteps", len(scheduler.timesteps))
        )
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if bool(((rounded <= 0) | (rounded > num_train_timesteps)).any()):
            raise ValueError(
                "formal warped Wan denoising steps must lie in "
                f"[1, {num_train_timesteps}], got {rounded.tolist()}"
            )
        scheduler_timesteps = scheduler.timesteps.detach().cpu()
        if len(scheduler_timesteps) != num_train_timesteps:
            raise ValueError(
                "Wan scheduler grid length does not match num_train_timesteps: "
                f"grid={len(scheduler_timesteps)} configured={num_train_timesteps}"
            )
        indices = num_train_timesteps - rounded.to(dtype=torch.long)
        resolved = scheduler_timesteps.index_select(0, indices)
    else:
        resolved = values.to(dtype=scheduler.timesteps.dtype)

    if device is not None:
        resolved = resolved.to(device=device)
    return resolved


__all__ = ["resolve_wan_denoising_steps"]
