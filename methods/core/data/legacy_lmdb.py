"""Read-only contract for historical latent LMDB datasets.

This module preserves the existing ``<array>_shape`` and
``<array>_<row>_data`` layout.  It does not define or recognize the retired
ODE trajectory format as a new protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from methods.core.errors import LegacyDataError


LEGACY_LATENT_LMDB_FORMAT = "legacy_latent_lmdb"


@dataclass(frozen=True)
class LegacyLmdbSummary:
    format: str
    length: int
    latents_shape: tuple[int, ...]
    prompts_shape: tuple[int, ...] | None


def shape_key(array_name: str) -> bytes:
    if not array_name or any(character.isspace() for character in array_name):
        raise LegacyDataError(f"invalid legacy LMDB array name: {array_name!r}")
    return f"{array_name}_shape".encode("utf-8")


def row_key(array_name: str, row_index: int) -> bytes:
    if row_index < 0:
        raise LegacyDataError(f"legacy LMDB row index must be non-negative: {row_index}")
    shape_key(array_name)
    return f"{array_name}_{row_index}_data".encode("utf-8")


def read_shape(env: Any, array_name: str) -> tuple[int, ...]:
    try:
        with env.begin() as transaction:
            raw = transaction.get(shape_key(array_name))
    except Exception as exc:
        raise LegacyDataError(f"cannot read {array_name!r} shape from legacy LMDB: {exc}") from exc
    if raw is None:
        raise LegacyDataError(f"legacy LMDB is missing key {shape_key(array_name)!r}")
    try:
        shape = tuple(int(item) for item in raw.decode("utf-8").split())
    except (UnicodeDecodeError, ValueError) as exc:
        raise LegacyDataError(f"invalid shape metadata for {array_name!r}: {raw!r}") from exc
    if not shape or any(dimension < 0 for dimension in shape):
        raise LegacyDataError(f"invalid shape metadata for {array_name!r}: {shape!r}")
    return shape


def read_row_bytes(env: Any, array_name: str, row_index: int) -> bytes:
    key = row_key(array_name, row_index)
    try:
        with env.begin() as transaction:
            value = transaction.get(key)
    except Exception as exc:
        raise LegacyDataError(f"cannot read legacy LMDB row {key!r}: {exc}") from exc
    if value is None:
        raise LegacyDataError(f"legacy LMDB is missing row key {key!r}")
    return bytes(value)


def read_prompt(env: Any, row_index: int) -> str:
    raw = read_row_bytes(env, "prompts", row_index)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LegacyDataError(f"prompt row {row_index} is not UTF-8") from exc


def read_latent(env: Any, row_index: int, *, dtype: Any = None) -> Any:
    """Return a NumPy view with the historical row shape."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - NumPy is a project dependency.
        raise LegacyDataError("NumPy is required to decode legacy latent LMDB rows") from exc
    selected_dtype = np.float16 if dtype is None else dtype
    shape = read_shape(env, "latents")
    raw = read_row_bytes(env, "latents", row_index)
    try:
        return np.frombuffer(raw, dtype=selected_dtype).reshape(shape[1:])
    except ValueError as exc:
        raise LegacyDataError(
            f"latent row {row_index} byte length is incompatible with shape {shape[1:]}"
        ) from exc


def validate_legacy_lmdb(env: Any, *, check_first_row: bool = True) -> LegacyLmdbSummary:
    latents_shape = read_shape(env, "latents")
    try:
        with env.begin() as transaction:
            raw_prompts_shape = transaction.get(shape_key("prompts"))
    except Exception as exc:
        raise LegacyDataError(f"cannot inspect prompt shape in legacy LMDB: {exc}") from exc
    # Some early datasets did not persist prompt shape even though prompt row
    # keys are present.  Preserve only that absence; malformed metadata still
    # fails validation.
    prompts_shape = None if raw_prompts_shape is None else read_shape(env, "prompts")
    length = latents_shape[0]
    if prompts_shape is not None and prompts_shape[0] != length:
        raise LegacyDataError(
            f"legacy LMDB length mismatch: latents={length}, prompts={prompts_shape[0]}"
        )
    if check_first_row and length:
        read_row_bytes(env, "latents", 0)
        read_prompt(env, 0)
    return LegacyLmdbSummary(
        format=LEGACY_LATENT_LMDB_FORMAT,
        length=length,
        latents_shape=latents_shape,
        prompts_shape=prompts_shape,
    )


def open_legacy_lmdb(path: str | Path) -> Any:
    """Open an existing LMDB read-only without making ``lmdb`` an import-time dependency."""

    try:
        import lmdb
    except ImportError as exc:
        raise LegacyDataError("the 'lmdb' package is required to open a legacy dataset") from exc
    candidate = Path(path)
    if not candidate.exists():
        raise LegacyDataError(f"legacy LMDB path does not exist: {candidate}")
    try:
        return lmdb.open(
            str(candidate),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    except Exception as exc:
        raise LegacyDataError(f"cannot open legacy LMDB {candidate}: {exc}") from exc


__all__ = [
    "LEGACY_LATENT_LMDB_FORMAT",
    "LegacyLmdbSummary",
    "open_legacy_lmdb",
    "read_latent",
    "read_prompt",
    "read_row_bytes",
    "read_shape",
    "row_key",
    "shape_key",
    "validate_legacy_lmdb",
]
