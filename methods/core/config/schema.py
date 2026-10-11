"""Structured configuration schemas.

Only the stable top-level interface is structured.  Stage-specific values are
kept under ``data``, ``artifacts`` and ``params`` so a new model family can add
its own settings without changing the model-independent package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


CONFIG_SCHEMA_VERSION = 1


@dataclass
class RecipeConfig:
    """Strict public interface for a stage recipe or model config layer."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    model_id: str = MISSING
    stage: str = MISSING
    temporal_mode: str = "chunkwise"
    roles: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig(RecipeConfig):
    """Resolved recipe with its separately loaded runtime profile."""

    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeFile:
    """Schema for a standalone runtime profile."""

    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelManifest:
    """Stable fields accepted in ``manifest.yaml``.

    ``defaults`` remains a namespaced mapping.  The loader understands
    ``recipe_dir`` and ``runtime`` directly and can consume optional layered
    config paths without widening the manifest's top-level API.
    """

    id: str = MISSING
    family: str = MISSING
    scale: str = MISSING
    adapter: str = MISSING
    capabilities: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ExperimentConfig",
    "ModelManifest",
    "RecipeConfig",
    "RuntimeFile",
]
