"""The model-independent five-stage lifecycle."""

from methods.core.stages.adaptive_dmd import DESCRIPTOR as ADAPTIVE_DMD
from methods.core.stages.base import ALL_ACTIONS, LifecycleAction, LifecycleDescriptor
from methods.core.stages.cd import DESCRIPTOR as CD
from methods.core.stages.dmd import DESCRIPTOR as DMD
from methods.core.stages.router import DESCRIPTOR as ROUTER
from methods.core.stages.tf import DESCRIPTOR as TF

DESCRIPTORS = (TF, CD, DMD, ADAPTIVE_DMD, ROUTER)

__all__ = [
    "ADAPTIVE_DMD",
    "ALL_ACTIONS",
    "CD",
    "DESCRIPTORS",
    "DMD",
    "LifecycleAction",
    "LifecycleDescriptor",
    "ROUTER",
    "TF",
]
