"""Distribution-matching-distillation lifecycle descriptor."""

from methods.core.stages.base import LifecycleDescriptor

DESCRIPTOR = LifecycleDescriptor(
    name="dmd",
    parent_stage="cd",
    required_roles=("generator", "real_teacher", "fake_score"),
)

__all__ = ["DESCRIPTOR"]
