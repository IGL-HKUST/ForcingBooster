"""Small, model-independent output interface for inference runs.

Tracked adapters intentionally default to :class:`VideoArtifactWriter`, which
only emits MP4 files.  Local experiment packages may supply another writer to
persist latents, router decisions, timing, or evaluation manifests without
putting those policies back into the model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class InferenceSample:
    """One decoded sample plus optional in-memory experiment diagnostics."""

    sample_id: str
    video: torch.Tensor
    filename: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tensors: Mapping[str, torch.Tensor] = field(default_factory=dict)


@runtime_checkable
class InferenceArtifactWriter(Protocol):
    """Public hook for choosing which inference artifacts are persisted."""

    def begin(self, output_dir: Path, metadata: Mapping[str, Any]) -> None: ...

    def write(self, output_dir: Path, sample: InferenceSample) -> Path: ...

    def finish(self, output_dir: Path, metadata: Mapping[str, Any]) -> None: ...


class VideoArtifactWriter:
    """Default writer: create the destination and write MP4 files only."""

    def __init__(
        self,
        *,
        fps: int = 16,
        crf: int | None = 18,
        quantization: str = "round",
    ) -> None:
        self.fps = int(fps)
        self.crf = None if crf is None else int(crf)
        if quantization not in {"round", "truncate"}:
            raise ValueError("quantization must be 'round' or 'truncate'")
        self.quantization = quantization

    def begin(self, output_dir: Path, metadata: Mapping[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, output_dir: Path, sample: InferenceSample) -> Path:
        from torchvision.io import write_video

        filename = sample.filename or f"{sample.sample_id}.mp4"
        destination = output_dir / filename
        video = sample.video.detach().cpu()
        if video.ndim != 4 or video.shape[-1] not in {1, 3, 4}:
            raise ValueError(
                "InferenceSample.video must be a THWC tensor, got "
                f"shape={tuple(video.shape)}"
            )
        if video.dtype != torch.uint8:
            video = video.clamp(0, 255)
            if self.quantization == "round":
                video = video.round()
            video = video.to(torch.uint8)
        kwargs: dict[str, Any] = {"video_codec": "libx264"}
        if self.crf is not None:
            kwargs["options"] = {"crf": str(self.crf)}
        write_video(str(destination), video, fps=self.fps, **kwargs)
        return destination

    def finish(self, output_dir: Path, metadata: Mapping[str, Any]) -> None:
        return None
