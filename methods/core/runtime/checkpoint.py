"""Versioned checkpoint artifact and legacy warm-start helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from methods.core.errors import CheckpointError
from methods.core.registry import get_stage


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FORMAT = "causal-forcing.checkpoint"
CHECKPOINT_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class CheckpointManifest:
    model_id: str
    stage: str
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    format: str = CHECKPOINT_FORMAT
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    model_file: str = "model.pt"
    training_state_file: str = "training_state.pt"
    resolved_config_file: str = "resolved_config.yaml"
    parent_checkpoint: str | None = None
    parent_digest: str | None = None
    save_full_generator: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegacyCheckpointSelection:
    state_dict: Mapping[str, Any]
    source_key: str
    resume_mode: str
    has_training_state: bool
    warnings: tuple[str, ...] = ()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(f"cannot hash checkpoint artifact {path}: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def validate_checkpoint_manifest(manifest: CheckpointManifest) -> None:
    if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema_version={manifest.schema_version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if manifest.format != CHECKPOINT_FORMAT:
        raise CheckpointError(f"unsupported checkpoint format {manifest.format!r}")
    if not isinstance(manifest.model_id, str) or not manifest.model_id.strip():
        raise CheckpointError("checkpoint model_id must not be empty")
    if not isinstance(manifest.stage, str):
        raise CheckpointError("checkpoint stage must be a string")
    get_stage(manifest.stage)
    for field_name in ("model_file", "training_state_file", "resolved_config_file"):
        value = getattr(manifest, field_name)
        if not isinstance(value, str):
            raise CheckpointError(f"{field_name} must be a string")
        path = Path(value)
        if not value or path == Path(".") or path.is_absolute() or ".." in path.parts:
            raise CheckpointError(f"{field_name} must be a checkpoint-relative path: {value!r}")
    if manifest.parent_digest is not None and not manifest.parent_digest.startswith("sha256:"):
        raise CheckpointError("parent_digest must use the 'sha256:<hex>' form")


def write_checkpoint_manifest(
    checkpoint_dir: str | Path,
    manifest: CheckpointManifest,
) -> Path:
    """Atomically write ``manifest.json`` after validating the contract."""

    validate_checkpoint_manifest(manifest)
    directory = Path(checkpoint_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / CHECKPOINT_MANIFEST_NAME
        temporary = directory / f".{CHECKPOINT_MANIFEST_NAME}.tmp"
        temporary.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target
    except OSError as exc:
        raise CheckpointError(f"cannot write checkpoint manifest below {directory}: {exc}") from exc


def load_checkpoint_manifest(checkpoint_dir: str | Path) -> CheckpointManifest:
    path = Path(checkpoint_dir)
    if path.is_dir() or path.name != CHECKPOINT_MANIFEST_NAME:
        path = path / CHECKPOINT_MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("root value must be an object")
        manifest = CheckpointManifest(**payload)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise CheckpointError(f"invalid checkpoint manifest {path}: {exc}") from exc
    validate_checkpoint_manifest(manifest)
    return manifest


def validate_checkpoint_directory(
    checkpoint_dir: str | Path,
    *,
    require_training_state: bool = True,
) -> CheckpointManifest:
    """Validate the manifest and required files of a new-format checkpoint."""

    directory = Path(checkpoint_dir)
    manifest = load_checkpoint_manifest(directory)
    required = [manifest.model_file, manifest.resolved_config_file]
    if require_training_state:
        required.append(manifest.training_state_file)
    missing = [relative for relative in required if not (directory / relative).is_file()]
    if missing:
        raise CheckpointError(
            f"checkpoint {directory} is missing required artifacts: {', '.join(missing)}"
        )
    return manifest


_WRAPPER_SEGMENTS = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_orig_mod.",
)


def _tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _state_dict_like(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and _tensor_like(item) for key, item in value.items())
    )


def normalize_legacy_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for original_key, value in state_dict.items():
        key = str(original_key)
        for segment in _WRAPPER_SEGMENTS:
            key = key.replace(segment, "")
        normalized[key] = value
    while normalized and all(key.startswith("generator.") for key in normalized):
        normalized = {key[len("generator.") :]: value for key, value in normalized.items()}
    return normalized


def select_legacy_state_dict(
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool = False,
) -> LegacyCheckpointSelection:
    """Select legacy ``generator``/``generator_ema``/bare Wan weights.

    Legacy files without optimizer state are deliberately classified as a
    warm start.  Callers must not treat them as strict training resumption.
    """

    if not isinstance(checkpoint, Mapping):
        raise CheckpointError(
            f"legacy checkpoint must be a mapping, got {type(checkpoint).__name__}"
        )
    source_key: str | None = None
    selected: Mapping[str, Any] | None = None
    if _state_dict_like(checkpoint):
        source_key, selected = "root", checkpoint
    else:
        order = ("generator_ema", "generator", "model") if use_ema else (
            "generator",
            "model",
            "generator_ema",
        )
        for key in order:
            candidate = checkpoint.get(key)
            if _state_dict_like(candidate):
                source_key, selected = key, candidate
                break
    if selected is None or source_key is None:
        raise CheckpointError(
            "legacy checkpoint contains no tensor-like generator state dict under "
            "root/generator/generator_ema/model"
        )

    has_optimizer_hint = any(
        key in checkpoint
        for key in ("optimizer", "optimizer_state", "optimizer_state_dict", "optimizers")
    )
    has_step_hint = any(key in checkpoint for key in ("step", "global_step", "train_step"))
    has_training_state = has_optimizer_hint and has_step_hint
    warnings = (
        "unversioned legacy checkpoints are warm-start only; stage-specific strict-resume "
        "state cannot be established from optimizer/step key names",
    )
    return LegacyCheckpointSelection(
        state_dict=normalize_legacy_state_dict(selected),
        source_key=source_key,
        resume_mode="warm_start",
        has_training_state=has_training_state,
        warnings=warnings,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_MANIFEST_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManifest",
    "LegacyCheckpointSelection",
    "load_checkpoint_manifest",
    "normalize_legacy_state_dict",
    "select_legacy_state_dict",
    "sha256_file",
    "validate_checkpoint_directory",
    "validate_checkpoint_manifest",
    "write_checkpoint_manifest",
]
