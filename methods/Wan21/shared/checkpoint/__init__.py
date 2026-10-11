"""Wan 2.1 checkpoint compatibility layer."""

from .legacy import (
    GeneratorLoadReport,
    load_generator_checkpoint,
    normalize_generator_state_dict,
    select_generator_state_dict,
)
from .io import load_checkpoint_bundle, save_checkpoint_bundle, save_generator_checkpoint

__all__ = [
    "GeneratorLoadReport",
    "load_generator_checkpoint",
    "normalize_generator_state_dict",
    "select_generator_state_dict",
    "load_checkpoint_bundle",
    "save_checkpoint_bundle",
    "save_generator_checkpoint",
]
