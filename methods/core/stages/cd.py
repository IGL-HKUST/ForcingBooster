"""Consistency-distillation lifecycle descriptor."""

from methods.core.stages.base import LifecycleDescriptor

DESCRIPTOR = LifecycleDescriptor(
    name="cd",
    parent_stage="tf",
    required_roles=("generator", "real_teacher"),
)

__all__ = ["DESCRIPTOR"]
