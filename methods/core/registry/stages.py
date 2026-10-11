"""Registry for the five supported training stages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from methods.core.errors import UnsupportedStageError
from methods.core.stages import DESCRIPTORS, LifecycleDescriptor


class StageRegistry(Mapping[str, LifecycleDescriptor]):
    """Immutable registry with an explicit ODE retirement error."""

    def __init__(self) -> None:
        self._descriptors = {descriptor.name: descriptor for descriptor in DESCRIPTORS}

    def __getitem__(self, stage: str) -> LifecycleDescriptor:
        normalized = normalize_stage(stage)
        try:
            return self._descriptors[normalized]
        except KeyError as exc:
            choices = ", ".join(self._descriptors)
            raise UnsupportedStageError(
                f"unsupported stage {stage!r}; supported stages: {choices}"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._descriptors)

    def __len__(self) -> int:
        return len(self._descriptors)


def normalize_stage(stage: str) -> str:
    normalized = str(stage).strip().lower().replace("-", "_")
    if normalized == "ode":
        raise UnsupportedStageError(
            "stage 'ode' was intentionally not migrated; use the Git history or an old branch "
            "to reproduce historical ODE experiments"
        )
    return normalized


STAGES = StageRegistry()


def get_stage(stage: str) -> LifecycleDescriptor:
    return STAGES[stage]


def supported_stages() -> tuple[str, ...]:
    return tuple(STAGES)


__all__ = ["STAGES", "StageRegistry", "get_stage", "normalize_stage", "supported_stages"]
