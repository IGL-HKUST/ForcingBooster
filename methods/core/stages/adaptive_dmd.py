"""Adaptive-DMD lifecycle descriptor."""

from methods.core.stages.base import LifecycleDescriptor

DESCRIPTOR = LifecycleDescriptor(
    name="adaptive_dmd",
    parent_stage="dmd",
    required_roles=("generator", "real_teacher", "fake_score"),
)

__all__ = ["DESCRIPTOR"]
