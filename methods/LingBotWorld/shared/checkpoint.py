"""Final-only native router checkpoint storage and recovery."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from methods.core.runtime import (
    CheckpointManifest,
    load_checkpoint_manifest,
    sha256_file,
    validate_checkpoint_directory,
    write_checkpoint_manifest,
)


@dataclass(frozen=True)
class NativeRouterBundle:
    directory: Path
    manifest: CheckpointManifest
    model_state: dict[str, Any]
    training_state: dict[str, Any]


def _resolve_directory(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name == "manifest.json":
        candidate = candidate.parent
    return candidate.resolve(strict=False)


def resolve_checkpoint_kind(path: str | Path) -> str:
    directory = _resolve_directory(path)
    return "native_bundle" if (directory / "manifest.json").is_file() else "hf_revision"


def _update_latest(output_dir: Path, checkpoint_dir: Path) -> None:
    latest = output_dir / "latest"
    temporary = output_dir / f".latest.{uuid.uuid4().hex}.tmp"
    temporary.symlink_to(checkpoint_dir.name, target_is_directory=True)
    if latest.exists() and not latest.is_symlink():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"refusing to replace non-symlink marker {latest}")
    os.replace(temporary, latest)


def save_final_router_bundle(
    *,
    output_dir: str | Path,
    model_id: str,
    step: int,
    final_step: int,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    fsdp_topology: Mapping[str, Any],
    seed: int,
    router_runtime: Mapping[str, Any] | None = None,
    resume_checkpoint: str | Path | None = None,
    overwrite_final: bool = False,
    training_contract: Mapping[str, Any] | None = None,
    attempted_iterations: int | None = None,
) -> Path:
    """Atomically write the only checkpoint produced by one router invocation.

    A fresh invocation keeps the original single-checkpoint directory contract.
    A resumed invocation may retain exactly its immutable, earlier source bundle
    beside the new final bundle.  This makes the resume chain explicit without
    allowing unrelated or later checkpoints to accumulate silently.
    """

    step = int(step)
    final_step = int(final_step)
    if step != final_step:
        raise ValueError(
            f"final-only policy rejects checkpoint at step {step}; expected {final_step}"
        )
    if not model_state:
        raise ValueError("full generator state must not be empty")
    if not optimizer_state:
        raise ValueError("router optimizer state must not be empty")

    destination = _resolve_directory(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / f"checkpoint_step_{step:06d}"
    existing = sorted(destination.glob("checkpoint_step_*"))
    resume_directory = None
    resume_manifest = None
    resume_step = None
    if resume_checkpoint is not None:
        resume_directory = _resolve_directory(resume_checkpoint)
        if resume_directory.parent != destination:
            raise ValueError(
                "resume checkpoint must be an existing child of the router output directory"
            )
        resume_manifest = validate_checkpoint_directory(
            resume_directory, require_training_state=True
        )
        if resume_manifest.model_id != str(model_id) or resume_manifest.stage != "router":
            raise ValueError("resume checkpoint identity mismatch")
        resume_step = int(resume_manifest.metadata.get("step", -1))
        if resume_directory.name != f"checkpoint_step_{resume_step:06d}":
            raise ValueError("resume checkpoint directory name and manifest step disagree")
        if resume_step >= final_step:
            raise ValueError(
                f"resume checkpoint step {resume_step} must be earlier than final step {final_step}"
            )
    allowed_existing = {resume_directory} if resume_directory is not None else set()
    foreign = [path for path in existing if path not in allowed_existing and path != checkpoint_dir]
    if foreign:
        raise RuntimeError(
            "final-only output already contains additional checkpoints: "
            + ", ".join(path.name for path in foreign)
        )
    if checkpoint_dir.exists():
        if resume_directory is not None or not overwrite_final:
            raise FileExistsError(f"final checkpoint already exists: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    temporary = destination / f".{checkpoint_dir.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=False)
    try:
        torch.save({"generator": dict(model_state)}, temporary / "model.pt")
        torch.save(
            {
                "step": step,
                "router_optimizer": dict(optimizer_state),
                "seed": int(seed),
                "router_training_contract": dict(training_contract or {}),
                "attempted_iterations": step if attempted_iterations is None else attempted_iterations,
            },
            temporary / "training_state.pt",
        )
        OmegaConf.save(OmegaConf.create(dict(resolved_config)), temporary / "resolved_config.yaml")
        manifest = CheckpointManifest(
            model_id=str(model_id),
            stage="router",
            parent_checkpoint=(
                None if resume_directory is None else str(resume_directory)
            ),
            parent_digest=(
                None
                if resume_directory is None
                else sha256_file(resume_directory / "manifest.json")
            ),
            save_full_generator=True,
            metadata={
                "step": step,
                "final_checkpoint_only": True,
                "resume_mode": "stateful_non_deterministic",
                "fsdp_topology": dict(fsdp_topology),
                "model_state_keys": ["generator"],
                "training_state_keys": ["router_optimizer", "seed", "step",
                                        "router_training_contract", "attempted_iterations"],
                "resume_from_step": resume_step,
                "router_runtime": dict(router_runtime or {}),
            },
        )
        write_checkpoint_manifest(temporary, manifest)
        os.replace(temporary, checkpoint_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _update_latest(destination, checkpoint_dir)
    return checkpoint_dir


def load_native_router_bundle(path: str | Path, *, load_training_state: bool = True) -> NativeRouterBundle:
    directory = _resolve_directory(path)
    manifest = validate_checkpoint_directory(directory, require_training_state=load_training_state)
    if manifest.stage != "router" or not manifest.save_full_generator:
        raise ValueError("LingBot native checkpoint must be a full router bundle")
    if manifest.metadata.get("final_checkpoint_only") is not True:
        raise ValueError("LingBot router bundle is missing final-only provenance")
    model_state = torch.load(
        directory / manifest.model_file, map_location="cpu", weights_only=False
    )
    training_state = torch.load(
        directory / manifest.training_state_file, map_location="cpu", weights_only=False
    ) if load_training_state else {}
    if set(model_state) != {"generator"} or not isinstance(model_state["generator"], dict):
        raise ValueError("LingBot router model.pt must contain one full generator mapping")
    required = {"step", "router_optimizer", "seed"} if load_training_state else set()
    missing = required.difference(training_state)
    if missing:
        raise ValueError(f"LingBot router training state is missing {sorted(missing)}")
    return NativeRouterBundle(
        directory=directory,
        manifest=load_checkpoint_manifest(directory),
        model_state=model_state,
        training_state=training_state,
    )


__all__ = [
    "NativeRouterBundle",
    "load_native_router_bundle",
    "resolve_checkpoint_kind",
    "save_final_router_bundle",
]


def load_router_weights(path):
    """Load full generator weights without requiring optimizer state or a manifest."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "model.pt"
    state = torch.load(candidate, map_location="cpu", weights_only=False)
    if set(state) != {"generator"} or not isinstance(state["generator"], dict):
        raise ValueError("LingBot model.pt must contain one full generator mapping")
    return state["generator"]
