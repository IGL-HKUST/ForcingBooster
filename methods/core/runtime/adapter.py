"""Lazy model-adapter loading and lifecycle dispatch."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from omegaconf import DictConfig

from methods.core.config import LoadedConfig, ModelManifest
from methods.core.errors import AdapterError
from methods.core.registry import get_stage


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimum direct-dispatch adapter protocol.

    Adapters may implement only the lifecycle actions they support. Dispatch
    reports a clear error when a requested action is absent.
    """

    def prepare(self, config: DictConfig) -> Any: ...

    def train(self, config: DictConfig) -> Any: ...

    def infer(self, config: DictConfig) -> Any: ...

@dataclass(frozen=True)
class AdapterInvocation:
    action: str
    model_id: str
    stage: str
    result: Any


def _import_target(reference: str) -> Any:
    module_name, separator, attribute_path = reference.partition(":")
    if not separator or not module_name or not attribute_path:
        raise AdapterError(
            f"adapter reference must use 'module:attribute', got {reference!r}"
        )
    try:
        target: Any = importlib.import_module(module_name)
        for part in attribute_path.split("."):
            target = getattr(target, part)
        return target
    except (ImportError, AttributeError) as exc:
        raise AdapterError(f"cannot import adapter {reference!r}: {exc}") from exc


def _accepts_keyword(callable_object: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


def load_adapter(reference: str, manifest: ModelManifest) -> Any:
    """Import and instantiate an adapter only after config validation."""

    target = _import_target(reference)
    if not inspect.isclass(target) and any(
        callable(getattr(target, action, None)) for action in ("prepare", "train", "infer", "validate")
    ):
        return target
    if not callable(target):
        raise AdapterError(f"adapter target is neither an adapter nor a factory: {reference!r}")
    try:
        if _accepts_keyword(target, "manifest"):
            return target(manifest=manifest)
        return target()
    except Exception as exc:
        raise AdapterError(f"adapter factory {reference!r} failed: {exc}") from exc


def _stage_runner(adapter: Any, loaded: LoadedConfig) -> Any:
    factory = getattr(adapter, "create_stage", None)
    if factory is None:
        return adapter
    if not callable(factory):
        raise AdapterError("adapter.create_stage exists but is not callable")
    descriptor = get_stage(str(loaded.config.stage))
    kwargs = {
        "stage": descriptor.name,
        "descriptor": descriptor,
        "config": loaded.config,
        "manifest": loaded.model.manifest,
    }
    try:
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(factory).parameters.values()
        ):
            return factory(**kwargs)
        accepted = {
            name: value
            for name, value in kwargs.items()
            if _accepts_keyword(factory, name)
        }
        return factory(**accepted)
    except Exception as exc:
        raise AdapterError(
            f"adapter failed to create stage runner for {descriptor.name!r}: {exc}"
        ) from exc


def run_lifecycle(adapter: Any, action: str, loaded: LoadedConfig) -> AdapterInvocation:
    descriptor = get_stage(str(loaded.config.stage))
    normalized_action = str(action).lower()
    if not descriptor.supports(normalized_action):
        raise AdapterError(
            f"stage {descriptor.name!r} does not declare lifecycle action {normalized_action!r}"
        )
    runner = _stage_runner(adapter, loaded)
    method = getattr(runner, normalized_action, None)
    if not callable(method):
        raise AdapterError(
            f"adapter runner {type(runner).__name__} does not implement {normalized_action}()"
        )
    try:
        if _accepts_keyword(method, "config"):
            result = method(config=loaded.config)
        else:
            result = method(loaded.config)
    except Exception as exc:
        raise AdapterError(
            f"{loaded.model.id}/{descriptor.name} lifecycle {normalized_action} failed: {exc}"
        ) from exc
    return AdapterInvocation(
        action=normalized_action,
        model_id=loaded.model.id,
        stage=descriptor.name,
        result=result,
    )


def load_and_run(action: str, loaded: LoadedConfig) -> AdapterInvocation:
    adapter = load_adapter(loaded.model.manifest.adapter, loaded.model.manifest)
    return run_lifecycle(adapter, action, loaded)


__all__ = [
    "AdapterInvocation",
    "ModelAdapter",
    "load_adapter",
    "load_and_run",
    "run_lifecycle",
]
