"""Versioned Wan training-checkpoint writer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from methods.core.runtime import (
    CheckpointManifest,
    sha256_file,
    validate_checkpoint_directory,
    write_checkpoint_manifest,
)


def _artifact_metadata(config) -> dict[str, Any]:
    value = getattr(config, "artifact_manifest", {}) or {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


def _parent_model_file(config) -> Path | None:
    metadata = _artifact_metadata(config)
    value = metadata.get("parent_checkpoint")
    if value in (None, "", False):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_dir():
        path = path / "model.pt"
    return path.resolve()


def _update_latest(output_dir: Path, checkpoint_dir: Path) -> None:
    latest = output_dir / "latest"
    temporary = output_dir / ".latest.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint_dir.name, target_is_directory=True)
    if latest.exists() and not latest.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink checkpoint marker: {latest}")
    os.replace(temporary, latest)


def save_checkpoint_bundle(
    *,
    config,
    step: int,
    model_state: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> Path:
    """Save model, stateful-resume data, resolved config, and manifest."""

    output_dir = Path(str(config.logdir)).expanduser().resolve()
    checkpoint_dir = output_dir / f"checkpoint_step_{int(step):06d}"
    metadata = _artifact_metadata(config)
    parent = _parent_model_file(config)
    if parent is not None and not parent.is_file():
        raise FileNotFoundError(f"parent checkpoint model file does not exist: {parent}")
    parent_digest = sha256_file(parent) if parent is not None else None
    manifest_metadata = {
        "step": int(step),
        "resume_mode": "stateful_non_deterministic",
        "model_state_keys": sorted(model_state),
        "training_state_keys": sorted(training_state),
    }
    for key in ("router_runtime", "checkpoint_runtime_audit"):
        if metadata.get(key) is not None:
            manifest_metadata[key] = metadata[key]
    manifest = CheckpointManifest(
        model_id=str(metadata.get("model_id", getattr(config, "model_id", ""))),
        stage=str(metadata.get("stage", getattr(config, "stage", ""))),
        parent_checkpoint=(None if parent is None else str(parent)),
        parent_digest=parent_digest,
        save_full_generator=bool(metadata.get("save_full_generator", True)),
        metadata=manifest_metadata,
    )
    _validate_stage_payload(manifest, model_state, training_state)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state), checkpoint_dir / "model.pt")
    torch.save(dict(training_state), checkpoint_dir / "training_state.pt")
    resolved = getattr(config, "resolved_config", None)
    if resolved is None:
        resolved = OmegaConf.to_container(config, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), checkpoint_dir / "resolved_config.yaml")
    write_checkpoint_manifest(checkpoint_dir, manifest)
    _update_latest(output_dir, checkpoint_dir)
    return checkpoint_dir


def save_generator_checkpoint(
    *,
    config,
    step: int,
    generator_state: Mapping[str, Any],
) -> Path:
    """Save a generator-only warm-start artifact without optimizer state."""

    if not isinstance(generator_state, Mapping) or not generator_state:
        raise ValueError("generator-only checkpoint requires a non-empty generator state")

    output_dir = Path(str(config.logdir)).expanduser().resolve()
    checkpoint_dir = output_dir / f"checkpoint_step_{int(step):06d}"
    metadata = _artifact_metadata(config)
    parent = _parent_model_file(config)
    if parent is not None and not parent.is_file():
        raise FileNotFoundError(f"parent checkpoint model file does not exist: {parent}")
    parent_digest = sha256_file(parent) if parent is not None else None
    manifest_metadata = {
        "step": int(step),
        "resume_mode": "warm_start",
        "checkpoint_mode": "generator_only",
        "model_load_dtype": getattr(config, "model_load_dtype", None),
        "model_state_keys": ["generator"],
        "training_state_keys": [],
    }
    for key in ("router_runtime", "checkpoint_runtime_audit"):
        if metadata.get(key) is not None:
            manifest_metadata[key] = metadata[key]
    manifest = CheckpointManifest(
        model_id=str(metadata.get("model_id", getattr(config, "model_id", ""))),
        stage=str(metadata.get("stage", getattr(config, "stage", ""))),
        parent_checkpoint=(None if parent is None else str(parent)),
        parent_digest=parent_digest,
        save_full_generator=True,
        metadata=manifest_metadata,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": dict(generator_state)}, checkpoint_dir / "model.pt")
    resolved = getattr(config, "resolved_config", None)
    if resolved is None:
        resolved = OmegaConf.to_container(config, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), checkpoint_dir / "resolved_config.yaml")
    write_checkpoint_manifest(checkpoint_dir, manifest)
    _update_latest(output_dir, checkpoint_dir)
    return checkpoint_dir


def load_checkpoint_bundle(checkpoint: str | Path) -> tuple[CheckpointManifest, dict, dict]:
    """Load a checkpoint produced by :func:`save_checkpoint_bundle`."""

    directory = Path(checkpoint).expanduser().resolve()
    if directory.is_file():
        directory = directory.parent
    manifest = validate_checkpoint_directory(directory, require_training_state=True)
    model_state = torch.load(directory / manifest.model_file, map_location="cpu", weights_only=False)
    training_state = torch.load(
        directory / manifest.training_state_file,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(model_state, dict) or not isinstance(training_state, dict):
        raise TypeError(f"checkpoint bundle below {directory} must contain mapping payloads")
    _validate_stage_payload(manifest, model_state, training_state)
    return manifest, model_state, training_state


def _validate_stage_payload(
    manifest: CheckpointManifest,
    model_state: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> None:
    required_model = {"generator"}
    required_training = {"step"}
    if manifest.stage in {"tf", "cd"}:
        required_training.add("generator_optimizer")
    elif manifest.stage in {"dmd", "adaptive_dmd"}:
        required_model.add("fake_score")
        required_training.update({"generator_optimizer", "critic_optimizer"})
    elif manifest.stage == "router":
        required_training.add("router_optimizer")
        if not manifest.save_full_generator:
            raise ValueError("router checkpoint manifest must declare save_full_generator=true")

    missing_model = sorted(required_model.difference(model_state))
    missing_training = sorted(required_training.difference(training_state))
    if missing_model or missing_training:
        raise ValueError(
            "checkpoint payload is incomplete for stage "
            f"{manifest.stage!r}: missing model={missing_model} "
            f"training_state={missing_training}"
        )
    if not isinstance(model_state["generator"], Mapping) or not model_state["generator"]:
        raise ValueError("checkpoint generator state must be a non-empty mapping")


__all__ = [
    "load_checkpoint_bundle",
    "save_checkpoint_bundle",
    "save_generator_checkpoint",
]
