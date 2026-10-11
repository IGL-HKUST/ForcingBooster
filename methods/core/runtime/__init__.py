"""Runtime adapter and checkpoint contracts."""

from methods.core.runtime.adapter import (
    AdapterInvocation,
    ModelAdapter,
    load_adapter,
    load_and_run,
    run_lifecycle,
)
from methods.core.runtime.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
    LegacyCheckpointSelection,
    load_checkpoint_manifest,
    normalize_legacy_state_dict,
    select_legacy_state_dict,
    sha256_file,
    validate_checkpoint_directory,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)

__all__ = [
    "AdapterInvocation",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_MANIFEST_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManifest",
    "LegacyCheckpointSelection",
    "ModelAdapter",
    "load_adapter",
    "load_and_run",
    "load_checkpoint_manifest",
    "normalize_legacy_state_dict",
    "run_lifecycle",
    "select_legacy_state_dict",
    "sha256_file",
    "validate_checkpoint_directory",
    "validate_checkpoint_manifest",
    "write_checkpoint_manifest",
]
