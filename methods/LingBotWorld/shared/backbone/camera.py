"""Backbone-facing wrappers for the shared LingBot camera representation.

Ray geometry has one numerical source in ``shared.utils.camera``.  This module
only supplies batching and the RGB-to-latent channel fold used by the official
Fast backbone.
"""

from __future__ import annotations

import torch

from methods.LingBotWorld.shared.utils.camera import camera_origin_direction_rays


def _intrinsics_four_vector(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.shape[-2:] == (3, 3):
        return torch.stack(
            (
                intrinsics[..., 0, 0],
                intrinsics[..., 1, 1],
                intrinsics[..., 0, 2],
                intrinsics[..., 1, 2],
            ),
            dim=-1,
        )
    if intrinsics.shape[-1:] == (4,):
        return intrinsics
    raise ValueError(f"unsupported intrinsics shape {tuple(intrinsics.shape)}")


def build_camera_origin_direction_rays(
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    *,
    use_first_intrinsics: bool = True,
) -> torch.Tensor:
    """Batch the shared ray builder and return ``[B, 6, F, H, W]``.

    The first three channels are camera-ray origins and the final three are
    unit directions.  LingBot's released inference uses the first frame's
    intrinsics for the whole clip, retained by default for checkpoint parity.
    """

    if c2ws.ndim == 3:
        c2ws = c2ws.unsqueeze(0)
    if c2ws.ndim != 4 or c2ws.shape[-2:] != (4, 4):
        raise ValueError(
            "c2ws must have shape [F,4,4] or [B,F,4,4], "
            f"got {tuple(c2ws.shape)}"
        )
    intrinsics = _intrinsics_four_vector(intrinsics)
    batch_size, num_frames = c2ws.shape[:2]
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.view(1, 1, 4)
    elif intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3:
        raise ValueError(
            "intrinsics must reduce to [4], [F,4], or [B,F,4], "
            f"got {tuple(intrinsics.shape)}"
        )
    if intrinsics.shape[0] not in (1, batch_size):
        raise ValueError(
            f"intrinsics batch dimension must be 1 or {batch_size}, got {intrinsics.shape[0]}"
        )
    if intrinsics.shape[1] not in (1, num_frames):
        raise ValueError(
            f"intrinsics frame dimension must be 1 or {num_frames}, got {intrinsics.shape[1]}"
        )
    intrinsics = intrinsics.to(device=c2ws.device, dtype=c2ws.dtype).expand(
        batch_size, num_frames, 4
    )
    if use_first_intrinsics:
        intrinsics = intrinsics[:, :1].expand_as(intrinsics)
    rays = [
        camera_origin_direction_rays(pose, intrinsic, height, width)
        for pose, intrinsic in zip(c2ws, intrinsics)
    ]
    return torch.stack(rays, dim=0).permute(0, 4, 1, 2, 3).contiguous()


def pack_camera_rays(
    camera_rays: torch.Tensor,
    *,
    spatial_stride: tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """Fold full-resolution rays into channels on the Wan latent grid."""

    if camera_rays.ndim == 4:
        camera_rays = camera_rays.unsqueeze(0)
    if camera_rays.ndim != 5 or camera_rays.shape[1] != 6:
        raise ValueError(
            "raw camera rays must have shape [B,6,F,H,W] or [6,F,H,W], "
            f"got {tuple(camera_rays.shape)}"
        )
    stride_h, stride_w = map(int, spatial_stride)
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"camera spatial stride must be positive, got {spatial_stride}")
    batch, channels, frames, height, width = camera_rays.shape
    if height % stride_h or width % stride_w:
        raise ValueError(
            f"camera ray resolution {(height, width)} is not divisible by stride {spatial_stride}"
        )
    return (
        camera_rays.view(
            batch,
            channels,
            frames,
            height // stride_h,
            stride_h,
            width // stride_w,
            stride_w,
        )
        .permute(0, 1, 4, 6, 2, 3, 5)
        .reshape(
            batch,
            channels * stride_h * stride_w,
            frames,
            height // stride_h,
            width // stride_w,
        )
        .contiguous()
    )


__all__ = ["build_camera_origin_direction_rays", "pack_camera_rays"]
