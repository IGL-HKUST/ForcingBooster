"""Lifecycle descriptors shared by every model family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LifecycleAction(str, Enum):
    PREPARE = "prepare"
    TRAIN = "train"
    INFER = "infer"
    VALIDATE = "validate"


ALL_ACTIONS = tuple(action.value for action in LifecycleAction)


@dataclass(frozen=True)
class LifecycleDescriptor:
    """Declarative contract for one stage.

    It deliberately contains no model implementation.  A concrete adapter is
    responsible for executing the declared lifecycle operations.
    """

    name: str
    parent_stage: str | None
    required_roles: tuple[str, ...]
    actions: tuple[str, ...] = ALL_ACTIONS
    temporal_mode: str = "chunkwise"

    def supports(self, action: str) -> bool:
        return str(action).lower() in self.actions


__all__ = ["ALL_ACTIONS", "LifecycleAction", "LifecycleDescriptor"]
