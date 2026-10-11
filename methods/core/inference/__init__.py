"""Stable inference extension points shared by model adapters."""

from .artifacts import InferenceArtifactWriter, InferenceSample, VideoArtifactWriter

__all__ = ["InferenceArtifactWriter", "InferenceSample", "VideoArtifactWriter"]
