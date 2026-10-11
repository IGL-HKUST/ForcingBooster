"""Resolve HF sources before model loading; local repositories stay untouched."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "ckpts" / "huggingface"
ASSET_PATTERNS = (
    "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
    "google/umt5-xxl/*.json", "google/umt5-xxl/*.model",
)
MODEL_PATTERNS = ("config.json", "*.safetensors", "*.safetensors.index.json")


def _source(value):
    if not value:
        raise ValueError("A weights source must be configured")
    value = str(value)
    local = Path(value).expanduser()
    local = local if local.is_absolute() else ROOT / local
    if local.exists():
        return local, None, None, "", False
    url = urlparse(value)
    if url.scheme:
        if url.scheme != "https" or url.netloc != "huggingface.co":
            raise ValueError(f"Expected a local path or https://huggingface.co source: {value}")
        parts = url.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"HF source must include owner/repository: {value}")
        repo, revision, prefix, is_file = "/".join(parts[:2]), "main", "", False
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in {"tree", "blob", "resolve"}:
                raise ValueError(f"Unsupported HF source URL: {value}")
            revision = unquote(parts[3])
            prefix = unquote("/".join(parts[4:]))
            is_file = parts[2] in {"blob", "resolve"}
            if is_file and not prefix:
                raise ValueError(f"HF file URL must include a filename: {value}")
        return None, repo, revision, prefix, is_file
    if (value.startswith(("/", ".", "~", "ckpts/", "logs/", "outputs/"))
            or not re.fullmatch(r"[\w.-]+/[\w.-]+", value)):
        raise FileNotFoundError(f"Local weights source does not exist: {local}")
    return None, value, "main", "", False


def _hub_kwargs(repo, revision):
    return dict(repo_id=repo, revision=revision, cache_dir=str(CACHE),
                endpoint="https://huggingface.co",
                local_files_only=os.environ.get("HF_HUB_OFFLINE", "").upper()
                in {"1", "ON", "YES", "TRUE"})


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required weights file or directory is missing: {path}")
    return path


def directory(source, *, patterns, required=()):
    """Fetch selected files and return the requested repository (sub)directory."""
    local, repo, revision, prefix, is_file = _source(source)
    if local is not None:
        root = local
    else:
        if is_file:
            raise ValueError(f"Expected a repository directory, got a file URL: {source}")
        from huggingface_hub import snapshot_download
        selected = [f"{prefix}/{p}" if prefix else p for p in patterns]
        root = Path(snapshot_download(allow_patterns=selected, **_hub_kwargs(repo, revision)))
        if prefix:
            root = root / prefix
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a weights repository directory: {root}")
    for name in required:
        require(root / name)
    return root


def checkpoint(source, *, filename="model.pt"):
    """Accept an explicit file, training bundle, or known file in a HF repo."""
    local, repo, revision, prefix, is_file = _source(source)
    filenames = (filename,) if isinstance(filename, str) else tuple(filename)
    if local is not None:
        if local.is_file():
            return local
        # Native training bundles always store model.pt at their root.
        for name in ("model.pt", *filenames):
            if (local / name).is_file():
                return local / name
        raise FileNotFoundError(f"Missing checkpoint; expected one of: {[str(local / p) for p in filenames]}")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    if is_file:
        filenames = (prefix,)
    for name in filenames:
        target = name if is_file or not prefix or name.startswith(prefix + "/") else f"{prefix}/{name}"
        try:
            return Path(hf_hub_download(filename=target, **_hub_kwargs(repo, revision)))
        except (EntryNotFoundError, LocalEntryNotFoundError):
            if name == filenames[-1]:
                raise


def validate_shards(root):
    """Validate a known model directory, including every indexed shard."""
    root = Path(root)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        for index in indexes:
            for name in set(json.loads(index.read_text())["weight_map"].values()):
                require(root / name)
    elif not list(root.glob("*.safetensors")):
        raise FileNotFoundError(f"No model safetensors or shard index in {root}")
    return root


def wan_model(source, *, assets=True):
    root = directory(source, patterns=MODEL_PATTERNS + (ASSET_PATTERNS if assets else ()),
                     required=("config.json",))
    validate_shards(root)
    if assets:
        validate_assets(root)
    return root


def validate_assets(root):
    for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
                 "google/umt5-xxl/tokenizer_config.json", "google/umt5-xxl/spiece.model"):
        require(Path(root) / name)
    return root


def wan_assets(source):
    return validate_assets(directory(source, patterns=ASSET_PATTERNS))
