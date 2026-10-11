"""Dependency-light adapter used by lifecycle contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True)
class DummyResult:
    action: str
    model_id: str
    stage: str
    configured: bool = True


class DummyAdapter:
    """Records all lifecycle calls without importing a concrete model family."""

    def __init__(self, manifest: Any = None) -> None:
        self.manifest = manifest
        self.calls: list[DummyResult] = []

    def _run(self, action: str, config: DictConfig) -> DummyResult:
        result = DummyResult(
            action=action,
            model_id=str(config.model_id),
            stage=str(config.stage),
        )
        self.calls.append(result)
        return result

    def prepare(self, config: DictConfig) -> DummyResult:
        return self._run("prepare", config)

    def train(self, config: DictConfig) -> DummyResult:
        return self._run("train", config)

    def infer(self, config: DictConfig) -> DummyResult:
        return self._run("infer", config)

    def validate(self, config: DictConfig) -> DummyResult:
        return self._run("validate", config)


__all__ = ["DummyAdapter", "DummyResult"]
