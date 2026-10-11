"""Chunkwise router lifecycle descriptor."""

from methods.core.stages.base import LifecycleDescriptor

DESCRIPTOR = LifecycleDescriptor(
    name="router",
    parent_stage="adaptive_dmd",
    required_roles=("generator",),
)

__all__ = ["DESCRIPTOR"]
