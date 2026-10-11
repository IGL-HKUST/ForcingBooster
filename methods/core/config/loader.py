"""Layered, structured experiment configuration loading."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from methods.core.config.schema import (
    CONFIG_SCHEMA_VERSION,
    ExperimentConfig,
    RecipeConfig,
    RuntimeFile,
)
from methods.core.errors import ConfigurationError
from methods.core.registry.models import ModelEntry, ModelRegistry
from methods.core.registry.stages import get_stage, normalize_stage


_LAYER_KEYS = ("common", "family", "shared", "profile", "model", "config")
_DEFAULT_CONTROL_KEYS = {"recipe", "recipe_dir", "runtime", *_LAYER_KEYS}


@dataclass(frozen=True)
class LoadedConfig:
    """Resolved config plus the artifact paths used to build it."""

    config: DictConfig
    model: ModelEntry
    recipe_path: Path
    runtime_path: Path | None
    layer_paths: tuple[Path, ...] = ()

    def to_yaml(self, *, resolve: bool = True) -> str:
        return OmegaConf.to_yaml(self.config, resolve=resolve, sort_keys=True)

    def as_dict(self, *, resolve: bool = True) -> dict[str, Any]:
        value = OmegaConf.to_container(self.config, resolve=resolve)
        assert isinstance(value, dict)
        return value


def _load_yaml_mapping(path: Path, description: str) -> DictConfig:
    try:
        value = OmegaConf.load(path)
    except (OSError, OmegaConfBaseException) as exc:
        raise ConfigurationError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, DictConfig):
        raise ConfigurationError(f"{description} must contain a YAML mapping: {path}")
    return value


def _strict_recipe_fragment(path: Path, description: str) -> DictConfig:
    raw = _load_yaml_mapping(path, description)
    try:
        # Merging against the structured schema rejects unknown top-level keys.
        OmegaConf.merge(OmegaConf.structured(RecipeConfig), raw)
    except OmegaConfBaseException as exc:
        raise ConfigurationError(f"invalid {description} {path}: {exc}") from exc
    return raw


def _runtime_fragment(path: Path) -> DictConfig:
    raw = _load_yaml_mapping(path, "runtime profile")
    try:
        checked = OmegaConf.merge(OmegaConf.structured(RuntimeFile), raw)
    except OmegaConfBaseException as exc:
        raise ConfigurationError(f"invalid runtime profile {path}: {exc}") from exc
    return OmegaConf.create({"runtime": checked.runtime})


def _paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return [str(item) for item in value]
    raise ConfigurationError(f"layer path must be a string or list of strings, got {value!r}")


def _resolve_existing_path(value: str, base: Path, description: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, base / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise ConfigurationError(f"{description} not found; checked: {rendered}")


def _resolve_named_yaml(
    value: str | Path,
    *,
    workspace: Path,
    directory: str | Path,
    description: str,
) -> Path:
    text = str(value)
    direct = Path(text).expanduser()
    direct_candidates = [direct] if direct.is_absolute() else [Path.cwd() / direct, workspace / direct]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()

    name = direct.name
    if not Path(name).suffix:
        name = f"{name}.yaml"
    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = workspace / directory_path
    candidate = directory_path / name
    if candidate.is_file():
        return candidate.resolve()
    raise ConfigurationError(f"{description} not found: {candidate}")


def _manifest_default_fragment(entry: ModelEntry, stage: str) -> DictConfig:
    defaults = dict(entry.manifest.defaults)
    fragment: dict[str, Any] = {
        "model_id": entry.id,
        "stage": stage,
    }
    if "temporal_mode" in defaults:
        fragment["temporal_mode"] = defaults["temporal_mode"]

    params: dict[str, Any] = {}
    explicit_params = defaults.get("params", {})
    if explicit_params:
        if not isinstance(explicit_params, Mapping):
            raise ConfigurationError("manifest defaults.params must be a mapping")
        params.update(explicit_params)
    for key, value in defaults.items():
        if key not in _DEFAULT_CONTROL_KEYS and key not in {"temporal_mode", "params"}:
            params[key] = value
    if params:
        fragment["params"] = params
    return OmegaConf.create(fragment)


def _recipe_path(entry: ModelEntry, stage: str, recipe: str | Path | None) -> Path:
    defaults = entry.manifest.defaults
    recipe_dir = str(defaults.get("recipe_dir", "config"))
    requested = recipe if recipe is not None else defaults.get("recipe", stage)
    if isinstance(requested, Mapping):
        requested = requested.get(stage, stage)
    if not isinstance(requested, (str, Path)):
        raise ConfigurationError(f"invalid recipe selector for stage {stage!r}: {requested!r}")
    return _resolve_named_yaml(
        requested,
        workspace=entry.workspace_dir,
        directory=recipe_dir,
        description="recipe",
    )


def _runtime_path(entry: ModelEntry, runtime: str | Path | None) -> Path | None:
    selected = runtime if runtime is not None else entry.manifest.defaults.get("runtime")
    if selected in (None, "", False):
        return None
    if not isinstance(selected, (str, Path)):
        raise ConfigurationError(f"invalid runtime selector: {selected!r}")
    return _resolve_named_yaml(
        selected,
        workspace=entry.workspace_dir,
        directory="runtime",
        description="runtime profile",
    )


def _layer_fragments(entry: ModelEntry) -> tuple[list[DictConfig], tuple[Path, ...]]:
    fragments: list[DictConfig] = []
    paths: list[Path] = []
    defaults = entry.manifest.defaults
    for key in _LAYER_KEYS:
        for value in _paths(defaults.get(key)):
            path = _resolve_existing_path(value, entry.workspace_dir, f"{key} config")
            fragments.append(_strict_recipe_fragment(path, f"{key} config"))
            paths.append(path)
    return fragments, tuple(paths)


def _override_fragment(overrides: Iterable[str]) -> DictConfig:
    items = list(overrides)
    for item in items:
        if "=" not in item:
            raise ConfigurationError(
                f"invalid --set value {item!r}; expected a dotted assignment such as params.lr=1e-5"
            )
    try:
        return OmegaConf.from_dotlist(items)
    except OmegaConfBaseException as exc:
        raise ConfigurationError(f"invalid command-line override: {exc}") from exc


def validate_config(
    config: DictConfig,
    entry: ModelEntry,
    registry: ModelRegistry | None = None,
    expected_stage: str | None = None,
) -> None:
    """Validate cross-file invariants before a concrete adapter is imported."""

    try:
        missing = OmegaConf.missing_keys(config)
        if missing:
            raise ConfigurationError(f"missing required configuration keys: {sorted(missing)}")

        if int(config.schema_version) != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported recipe schema_version={config.schema_version}; "
                f"expected {CONFIG_SCHEMA_VERSION}"
            )
        if str(config.model_id) != entry.id:
            raise ConfigurationError(
                f"recipe model_id {config.model_id!r} does not match requested model {entry.id!r}"
            )

        stage_name = normalize_stage(str(config.stage))
        if expected_stage is not None and stage_name != normalize_stage(expected_stage):
            raise ConfigurationError(
                f"recipe stage {stage_name!r} does not match requested stage "
                f"{normalize_stage(expected_stage)!r}"
            )
        descriptor = get_stage(stage_name)
        if stage_name not in entry.manifest.capabilities:
            raise ConfigurationError(
                f"model {entry.id!r} does not declare capability {stage_name!r}"
            )
        if str(config.temporal_mode) != descriptor.temporal_mode:
            raise ConfigurationError(
                f"stage {stage_name!r} only supports temporal_mode={descriptor.temporal_mode!r}; "
                f"got {config.temporal_mode!r}"
            )

        roles = dict(config.roles)
        absent_roles = [role for role in descriptor.required_roles if not str(roles.get(role, "")).strip()]
        if absent_roles:
            raise ConfigurationError(
                f"stage {stage_name!r} requires non-empty model roles: {', '.join(absent_roles)}"
            )
        generator_id = roles.get("generator")
        if generator_id != entry.id:
            raise ConfigurationError(
                f"roles.generator must match recipe model_id {entry.id!r}; got {generator_id!r}"
            )
        for tied_role in ("fake_score", "ema_teacher"):
            target = roles.get(tied_role)
            if target is not None and target != generator_id:
                raise ConfigurationError(
                    f"roles.{tied_role} must match roles.generator {generator_id!r}; got {target!r}"
                )
        if registry is not None:
            registered_ids = set(registry)
            unknown_role_targets = sorted(
                {str(target) for target in roles.values() if str(target) not in registered_ids}
            )
            if unknown_role_targets:
                raise ConfigurationError(
                    "model roles reference unregistered model IDs: "
                    + ", ".join(unknown_role_targets)
                )

        artifacts = dict(config.artifacts)
        actual_parent = artifacts.get("parent_stage")
        if descriptor.parent_stage is None:
            if actual_parent not in (None, "", False):
                raise ConfigurationError(
                    f"stage {stage_name!r} must not declare a parent stage; got {actual_parent!r}"
                )
        elif actual_parent != descriptor.parent_stage:
            raise ConfigurationError(
                f"stage {stage_name!r} must declare artifacts.parent_stage="
                f"{descriptor.parent_stage!r}; got {actual_parent!r}"
            )

        if stage_name == "router" and artifacts.get("save_full_generator") is not True:
            raise ConfigurationError("router must set artifacts.save_full_generator=true")

        data_format = str(config.data.get("format", "")).lower()
        if "ode" in data_format:
            raise ConfigurationError("ODE data formats are not part of the reorganized data protocol")
    except ConfigurationError:
        raise
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid resolved configuration: {exc}") from exc


def load_experiment_config(
    registry: ModelRegistry,
    model_id: str,
    stage: str,
    *,
    recipe: str | Path | None = None,
    runtime: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> LoadedConfig:
    """Load configuration in the documented precedence order.

    Precedence is: core schema, optional family/profile layers, manifest model
    defaults, stage recipe, runtime profile, then command-line overrides.
    """

    normalized_stage = normalize_stage(stage)
    entry = registry.require_capability(model_id, normalized_stage)
    selected_recipe = _recipe_path(entry, normalized_stage, recipe)
    selected_runtime = _runtime_path(entry, runtime)
    layer_fragments, layer_paths = _layer_fragments(entry)

    fragments: list[Any] = [
        OmegaConf.structured(ExperimentConfig),
        *layer_fragments,
        _manifest_default_fragment(entry, normalized_stage),
        _strict_recipe_fragment(selected_recipe, "recipe"),
    ]
    if selected_runtime is not None:
        fragments.append(_runtime_fragment(selected_runtime))
    fragments.append(_override_fragment(overrides))

    try:
        config = OmegaConf.merge(*fragments)
    except OmegaConfBaseException as exc:
        raise ConfigurationError(f"cannot merge experiment configuration: {exc}") from exc

    validate_config(config, entry, registry, normalized_stage)
    return LoadedConfig(
        config=config,
        model=entry,
        recipe_path=selected_recipe,
        runtime_path=selected_runtime,
        layer_paths=layer_paths,
    )


__all__ = ["LoadedConfig", "load_experiment_config", "validate_config"]
