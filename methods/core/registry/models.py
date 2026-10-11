"""Filesystem-backed model registry.

Discovery only parses manifests.  In particular, it never imports the adapter
target, so listing and validating models cannot initialize a Wan backbone (or
any future MiniMax/LingBotWorld backend).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from methods.core.config.schema import ModelManifest
from methods.core.errors import RegistryError, UnknownModelError
from methods.core.registry.stages import get_stage, normalize_stage


@dataclass(frozen=True)
class ModelEntry:
    manifest: ModelManifest
    manifest_path: Path
    workspace_dir: Path

    @property
    def id(self) -> str:
        return self.manifest.id

    def supports(self, stage: str) -> bool:
        normalized = normalize_stage(stage)
        return normalized in {normalize_stage(item) for item in self.manifest.capabilities}


def _default_methods_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_methods_root(path: str | Path | None) -> Path:
    if path is None:
        return _default_methods_root()
    candidate = Path(path).expanduser().resolve()
    if (candidate / "methods").is_dir():
        return candidate / "methods"
    return candidate


def _load_manifest(path: Path) -> ModelManifest:
    try:
        raw = OmegaConf.load(path)
        if not isinstance(raw, DictConfig):
            raise RegistryError(f"manifest must contain a YAML mapping: {path}")
        structured = OmegaConf.structured(ModelManifest)
        merged = OmegaConf.merge(structured, raw)
        OmegaConf.resolve(merged)
        manifest = OmegaConf.to_object(merged)
    except RegistryError:
        raise
    except (OSError, OmegaConfBaseException, ValueError, TypeError) as exc:
        raise RegistryError(f"invalid model manifest {path}: {exc}") from exc

    if not isinstance(manifest, ModelManifest):
        raise RegistryError(f"manifest did not resolve to ModelManifest: {path}")
    if not manifest.id.strip():
        raise RegistryError(f"manifest id must not be empty: {path}")
    if not manifest.adapter.strip() or ":" not in manifest.adapter:
        raise RegistryError(
            f"manifest adapter must use the 'module:attribute' form: {path}"
        )
    if not manifest.capabilities:
        raise RegistryError(f"manifest must declare at least one capability: {path}")
    normalized_capabilities: list[str] = []
    for capability in manifest.capabilities:
        descriptor = get_stage(capability)
        if descriptor.name not in normalized_capabilities:
            normalized_capabilities.append(descriptor.name)
    manifest.capabilities = normalized_capabilities
    return manifest


class ModelRegistry(Mapping[str, ModelEntry]):
    """Mapping from public model IDs to discovered workspace manifests."""

    def __init__(self, entries: Mapping[str, ModelEntry], methods_root: Path) -> None:
        self._entries = dict(entries)
        self.methods_root = methods_root

    @classmethod
    def discover(cls, methods_root: str | Path | None = None) -> "ModelRegistry":
        root = _as_methods_root(methods_root)
        if not root.is_dir():
            raise RegistryError(f"methods root does not exist: {root}")

        entries: dict[str, ModelEntry] = {}
        for manifest_path in sorted(root.glob("*/*/manifest.yaml")):
            manifest = _load_manifest(manifest_path)
            if manifest.id in entries:
                previous = entries[manifest.id].manifest_path
                raise RegistryError(
                    f"duplicate model id {manifest.id!r}: {previous} and {manifest_path}"
                )
            entries[manifest.id] = ModelEntry(
                manifest=manifest,
                manifest_path=manifest_path,
                workspace_dir=manifest_path.parent,
            )
        if not entries:
            raise RegistryError(f"no manifests found below {root}/*/*/manifest.yaml")
        return cls(entries, root)

    def __getitem__(self, model_id: str) -> ModelEntry:
        try:
            return self._entries[model_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries)) or "<none>"
            raise UnknownModelError(
                f"unknown model {model_id!r}; registered models: {available}"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._entries

    def require_capability(self, model_id: str, stage: str) -> ModelEntry:
        entry = self[model_id]
        normalized = normalize_stage(stage)
        if not entry.supports(normalized):
            capabilities = ", ".join(entry.manifest.capabilities)
            raise RegistryError(
                f"model {model_id!r} does not support stage {normalized!r}; "
                f"manifest capabilities: {capabilities}"
            )
        return entry


__all__ = ["ModelEntry", "ModelRegistry"]
