"""LingBot-specific CFHW random-number consumption policy."""

from __future__ import annotations

import torch


def randn_cfhw_as_ntchw(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
    draw_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Draw in upstream NCFTHW memory order, then expose NTCHW."""

    if reference.ndim != 5:
        raise ValueError(f"reference must be NTCHW, got {tuple(reference.shape)}")
    batch, frames, channels, height, width = reference.shape
    dtype = reference.dtype if draw_dtype is None else draw_dtype
    noise = torch.randn(
        (batch, channels, frames, height, width),
        generator=generator,
        device=reference.device,
        dtype=dtype,
    )
    return noise.to(dtype=reference.dtype).permute(0, 2, 1, 3, 4).contiguous()


def initial_lingbot_noise(
    shape: tuple[int, int, int, int, int],
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Generator]:
    """Consume and retain the official FP32 CFHW latent trajectory."""

    batch, frames, channels, height, width = map(int, shape)
    if compute_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise TypeError(f"unsupported LingBot compute dtype {compute_dtype}")
    reference = torch.empty(
        (batch, frames, channels, height, width),
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = randn_cfhw_as_ntchw(
        reference,
        generator=generator,
        draw_dtype=torch.float32,
    )
    return noise, generator


__all__ = ["initial_lingbot_noise", "randn_cfhw_as_ntchw"]
