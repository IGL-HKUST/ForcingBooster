"""Lightweight configurable scheduler for LingBot-World v2 Fast."""

from __future__ import annotations

from typing import Optional

import torch

from methods.Wan21.shared.adapter.scheduler import FlowMatchScheduler


class LingBotFastScheduler(FlowMatchScheduler):
    """Integer-timestep LingBot Fast schedule over a configurable grid."""

    def __init__(self, *, shift: float = 10.0, num_train_timesteps: int = 1000) -> None:
        if int(num_train_timesteps) <= 0:
            raise ValueError("num_train_timesteps must be positive")
        super().__init__(
            num_inference_steps=int(num_train_timesteps),
            num_train_timesteps=int(num_train_timesteps),
            shift=shift,
            sigma_max=(int(num_train_timesteps) - 1) / int(num_train_timesteps),
            sigma_min=0.0,
            extra_one_step=True,
        )

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        denoising_strength: float = 1.0,
        training: bool = False,
        *,
        shift: Optional[float] = None,
    ) -> None:
        if num_inference_steps is None:
            num_inference_steps = int(self.num_train_timesteps)
        if shift is not None:
            self.shift = float(shift)
        super().set_timesteps(
            num_inference_steps=int(num_inference_steps),
            denoising_strength=denoising_strength,
            training=training,
        )
        # Upstream FlowUniPC explicitly casts sigma*timestep_count to int64.
        self.timesteps = self.timesteps.to(dtype=torch.int64)

    def fast_timesteps(
        self,
        indices: tuple[int, ...] = (0, 250, 500, 750),
    ) -> torch.Tensor:
        indices = tuple(int(item) for item in indices)
        if not indices:
            raise ValueError("LingBot fast timestep indices must not be empty")
        if min(indices) < 0 or max(indices) >= len(self.timesteps):
            raise ValueError(
                f"schedule has {len(self.timesteps)} entries, cannot select {indices}"
            )
        return self.timesteps[
            torch.tensor(indices, dtype=torch.long, device=self.timesteps.device)
        ]


def resolve_timesteps(
    scheduler,
    *,
    indices: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Select the configured entries from the scheduler's timestep grid."""

    return scheduler.fast_timesteps(indices=indices).to(
        device=device, dtype=torch.long
    )


__all__ = ["LingBotFastScheduler", "resolve_timesteps"]
