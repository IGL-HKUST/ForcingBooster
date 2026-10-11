from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Mapping, Sequence

import torch

from methods.Wan21.shared.adapter.wan_cache import WanBackboneSpec


_WRAPPER_SEGMENTS = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")


@dataclass(frozen=True)
class GeneratorLoadReport:
    source_key: str
    target: str
    tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    backbone_spec: WanBackboneSpec


def _is_tensor_state_dict(value) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(torch.is_tensor(item) for item in value.values())


def select_generator_state_dict(checkpoint, use_ema: bool = False) -> tuple[Mapping[str, torch.Tensor], str]:
    if _is_tensor_state_dict(checkpoint):
        return checkpoint, "root"
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    preferred = ("generator_ema", "generator", "model") if use_ema else ("generator", "model", "generator_ema")
    for key in preferred:
        if key in checkpoint and _is_tensor_state_dict(checkpoint[key]):
            return checkpoint[key], key
    raise KeyError(
        "checkpoint does not contain a tensor generator state dict; "
        f"top-level keys={list(checkpoint)[:8]}"
    )


def normalize_generator_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for original_key, value in state_dict.items():
        key = str(original_key)
        for segment in _WRAPPER_SEGMENTS:
            key = key.replace(segment, "")
        normalized[key] = value

    # Some full-module checkpoints retain a leading generator namespace even
    # after their top-level container has been selected.
    while normalized and all(key.startswith("generator.") for key in normalized):
        normalized = {key[len("generator."):]: value for key, value in normalized.items()}
    return normalized


def _allowed_missing(key: str, prefixes: Sequence[str]) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in prefixes)


def load_generator_checkpoint(
    generator,
    checkpoint_or_path,
    *,
    use_ema: bool = False,
    allow_missing_prefixes: Sequence[str] = (),
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
    print_report: bool = True,
) -> GeneratorLoadReport:
    checkpoint = (
        torch.load(
            checkpoint_or_path,
            map_location=map_location,
            weights_only=False,
            mmap=mmap,
        )
        if isinstance(checkpoint_or_path, (str, bytes, PathLike))
        else checkpoint_or_path
    )
    selected, source_key = select_generator_state_dict(checkpoint, use_ema=use_ema)
    state_dict = normalize_generator_state_dict(selected)

    wrapper_keys = set(generator.state_dict())
    state_keys = set(state_dict)
    if any(key.startswith("model.") for key in state_keys) or state_keys.issubset(wrapper_keys):
        target_module = generator
        target_name = "generator"
        allowed_prefixes = tuple(allow_missing_prefixes)
    else:
        # Bare CausVid keys do not carry the WanDiffusionWrapper's `model.` namespace.
        target_module = generator.model
        target_name = "generator.model"
        allowed_prefixes = tuple(
            prefix[len("model."):] if prefix.startswith("model.") else prefix
            for prefix in allow_missing_prefixes
        )

    if allowed_prefixes:
        incompatible = target_module.load_state_dict(state_dict, strict=False)
        missing = tuple(key for key in incompatible.missing_keys if not _allowed_missing(key, allowed_prefixes))
        unexpected = tuple(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint load mismatch for {target_name}: "
                f"missing={list(missing)[:16]} unexpected={list(unexpected)[:16]}"
            )
        reported_missing = tuple(incompatible.missing_keys)
        reported_unexpected = tuple(incompatible.unexpected_keys)
    else:
        # Strict loading deliberately propagates shape, missing-key and unexpected-key errors.
        target_module.load_state_dict(state_dict, strict=True)
        reported_missing = ()
        reported_unexpected = ()

    spec = getattr(generator, "backbone_spec", None)
    if spec is None:
        spec = WanBackboneSpec.from_model(generator.model)
    report = GeneratorLoadReport(
        source_key=source_key,
        target=target_name,
        tensor_count=len(state_dict),
        missing_keys=reported_missing,
        unexpected_keys=reported_unexpected,
        backbone_spec=spec,
    )
    if print_report:
        print(
            "[checkpoint-load] "
            f"source={report.source_key} target={report.target} tensors={report.tensor_count} "
            f"missing={list(report.missing_keys)} unexpected={list(report.unexpected_keys)} "
            f"{spec.summary()}",
            flush=True,
        )
    return report
