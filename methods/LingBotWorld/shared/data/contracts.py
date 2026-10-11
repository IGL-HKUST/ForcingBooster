"""Data contracts for LingBot-World v2 image-to-video conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

import torch


VAE_TEMPORAL_STRIDE = 4


@dataclass(frozen=True)
class LingBotV2WindowSpec:
    """Temporal geometry for one DL3DV camera window."""

    source_fps: int
    target_fps: int
    source_frames: int
    video_frames: int
    latent_frames: int
    chunk_size: int = 4

    def __post_init__(self) -> None:
        if min(
            self.source_fps,
            self.target_fps,
            self.source_frames,
            self.video_frames,
            self.latent_frames,
            self.chunk_size,
        ) <= 0:
            raise ValueError("window dimensions and rates must be positive")
        source_span = (self.video_frames - 1) * self.source_fps
        if source_span % self.target_fps != 0:
            raise ValueError("target timestamps must land on a finite source-frame span")
        expected_source_frames = source_span // self.target_fps + 1
        if self.source_frames != expected_source_frames:
            raise ValueError(
                f"source_frames must be {expected_source_frames}, got {self.source_frames}"
            )
        if (self.video_frames - 1) % VAE_TEMPORAL_STRIDE != 0:
            raise ValueError("Wan VAE windows require video_frames = 4n + 1")
        expected_latent_frames = (
            (self.video_frames - 1) // VAE_TEMPORAL_STRIDE + 1
        )
        if self.latent_frames != expected_latent_frames:
            raise ValueError(
                f"latent_frames must be {expected_latent_frames}, got {self.latent_frames}"
            )
        if self.latent_frames % self.chunk_size != 0:
            raise ValueError("latent_frames must contain complete causal chunks")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def chunks(self) -> int:
        return self.latent_frames // self.chunk_size

    def target_source_times(self) -> torch.Tensor:
        return (
            torch.arange(self.video_frames, dtype=torch.float64)
            * self.source_fps
            / self.target_fps
        )


WINDOW_PROFILES = {
    "train": LingBotV2WindowSpec(
        source_fps=24,
        target_fps=16,
        source_frames=115,
        video_frames=77,
        latent_frames=20,
    ),
    "smoke": LingBotV2WindowSpec(
        source_fps=24,
        target_fps=16,
        source_frames=43,
        video_frames=29,
        latent_frames=8,
    ),
}


class LingBotV2I2VSample(TypedDict):
    """Raw rank-zero sample before T5 and VAE conditioning."""

    prompt: str
    first_frame: torch.Tensor
    camera_c2ws: torch.Tensor
    camera_intrinsics: torch.Tensor
    sample_metadata: dict[str, Any]


def make_i2v_encode_video(
    first_frame: torch.Tensor,
    video_frame_count: int,
) -> torch.Tensor:
    """Create the upstream VAE input: first RGB frame followed by zero frames."""
    if video_frame_count <= 0 or (video_frame_count - 1) % VAE_TEMPORAL_STRIDE != 0:
        raise ValueError("video_frame_count must be positive and equal to 4n + 1")
    if first_frame.ndim not in (3, 4) or first_frame.shape[-3] != 3:
        raise ValueError(
            "first_frame must have shape (3, H, W) or (B, 3, H, W), got "
            f"{tuple(first_frame.shape)}"
        )
    temporal_dimension = -3 if first_frame.ndim == 3 else -3
    first = first_frame.unsqueeze(temporal_dimension)
    output_shape = list(first.shape)
    output_shape[temporal_dimension] = video_frame_count
    video = torch.zeros(output_shape, device=first_frame.device, dtype=first_frame.dtype)
    index = [slice(None)] * video.ndim
    index[temporal_dimension] = slice(0, 1)
    video[tuple(index)] = first
    return video


def build_i2v_condition_mask(
    video_frame_count: int,
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the four mask channels used by the released Wan I2V conditioner."""
    if video_frame_count <= 0 or (video_frame_count - 1) % VAE_TEMPORAL_STRIDE != 0:
        raise ValueError("video_frame_count must be positive and equal to 4n + 1")
    if height <= 0 or width <= 0:
        raise ValueError("latent spatial dimensions must be positive")
    mask = torch.ones(
        (1, video_frame_count, height, width), device=device, dtype=dtype
    )
    mask[:, 1:] = 0
    mask = torch.cat(
        [
            mask[:, :1].repeat_interleave(VAE_TEMPORAL_STRIDE, dim=1),
            mask[:, 1:],
        ],
        dim=1,
    )
    latent_frames = (video_frame_count - 1) // VAE_TEMPORAL_STRIDE + 1
    return mask.view(
        1, latent_frames, VAE_TEMPORAL_STRIDE, height, width
    ).transpose(1, 2)[0]


def assemble_i2v_condition(
    encoded_video_latents: torch.Tensor,
    video_frame_count: int,
) -> torch.Tensor:
    """Concatenate the four-channel mask with a 16-channel VAE video latent.

    ``encoded_video_latents`` must be the VAE encoding of the tensor returned
    by :func:`make_i2v_encode_video`, not an independently encoded first frame.
    """
    if encoded_video_latents.ndim not in (4, 5):
        raise ValueError(
            "encoded_video_latents must have shape (16, F, H, W) or "
            f"(B, 16, F, H, W), got {tuple(encoded_video_latents.shape)}"
        )
    channel_dimension = 0 if encoded_video_latents.ndim == 4 else 1
    temporal_dimension = 1 if encoded_video_latents.ndim == 4 else 2
    if encoded_video_latents.shape[channel_dimension] != 16:
        raise ValueError("LingBot-World v2 expects 16 VAE latent channels")
    expected_latent_frames = (
        (video_frame_count - 1) // VAE_TEMPORAL_STRIDE + 1
    )
    if encoded_video_latents.shape[temporal_dimension] != expected_latent_frames:
        raise ValueError(
            "VAE temporal length does not match video_frame_count: "
            f"{encoded_video_latents.shape[temporal_dimension]} vs {expected_latent_frames}"
        )
    height, width = encoded_video_latents.shape[-2:]
    mask = build_i2v_condition_mask(
        video_frame_count,
        height,
        width,
        device=encoded_video_latents.device,
        dtype=encoded_video_latents.dtype,
    )
    if encoded_video_latents.ndim == 5:
        mask = mask.unsqueeze(0).expand(encoded_video_latents.shape[0], -1, -1, -1, -1)
    return torch.cat([mask, encoded_video_latents], dim=channel_dimension)


__all__ = [
    "LingBotV2I2VSample",
    "LingBotV2WindowSpec",
    "VAE_TEMPORAL_STRIDE",
    "WINDOW_PROFILES",
    "assemble_i2v_condition",
    "build_i2v_condition_mask",
    "make_i2v_encode_video",
]
