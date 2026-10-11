"""Teacher-forcing lifecycle descriptor."""

from methods.core.stages.base import LifecycleDescriptor

DESCRIPTOR = LifecycleDescriptor(name="tf", parent_stage=None, required_roles=("generator",))

__all__ = ["DESCRIPTOR"]
