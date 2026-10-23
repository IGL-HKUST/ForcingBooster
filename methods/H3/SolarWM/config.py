"""SolarWM recipe validation and base/Stage2/router/decoder weight resolution."""
import math
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

PREPARE_PARTS = ("text_encoder", "processor", "tokenizer", "vae", "audio_vae", "scheduler")
MODEL_PARTS = {"transformer", "text_encoder", "vae", "audio_vae"}
STAGE2_FILES = ("ema.pt", "COMPLETE.json", "checkpoint-manifest.json")


def validate(cfg):
    if cfg.mode not in ("native", "router"):
        raise ValueError("mode must be native or router")
    if cfg.forced_exit_step is not None:
        if cfg.forced_exit_step not in (1, 2, 3, 4) or cfg.mode != "router":
            raise ValueError("forced_exit_step requires mode=router and a step in 1..4")
    if not 0 <= cfg.exit_threshold <= 1:
        raise ValueError("exit_threshold must lie in [0,1]")
    if cfg.kv_commit not in ("wan", "full"):
        raise ValueError("kv_commit must be wan or full")
    if cfg.decoder not in ("taeh3", "vae"):
        raise ValueError("decoder must be taeh3 or vae")
    return cfg


def validate_train(cfg):
    t = validate(cfg).train
    if not uses_head(cfg):
        raise ValueError("H3 router training requires mode=router without forced_exit_step")
    if not t.manifest or not t.split:
        raise ValueError("train.manifest and train.split must name the training samples")
    if t.continue_cost not in ("four", "wan"):
        raise ValueError("train.continue_cost must be four or wan")
    if not t.target_chunks or any(j not in range(1, 10) for j in t.target_chunks):
        raise ValueError("train.target_chunks must be a non-empty list of chunks in 1..9")
    if not t.selected_steps or any(s not in (1, 2, 3) for s in t.selected_steps):
        raise ValueError("train.selected_steps must be a non-empty list of steps in 1..3")
    if not math.isfinite(float(t.nfe_weight)) or t.nfe_weight < 0:
        raise ValueError("train.nfe_weight must be finite and non-negative")
    if not t.learning_rate > 0 or (t.max_grad_norm is not None and not t.max_grad_norm > 0):
        raise ValueError("train.learning_rate and train.max_grad_norm (or null) must be positive")
    if not 0 <= t.max_updates <= t.max_attempts or t.save_every < 1:
        raise ValueError("train needs 0 <= max_updates <= max_attempts and save_every >= 1")
    if t.resume and t.init_head:
        raise ValueError("train.resume restores a run; train.init_head only initializes the head")
    return cfg


def uses_head(cfg):
    return cfg.mode == "router" and cfg.forced_exit_step is None


def github_file(source):
    """A local file, or https://github.com/<owner>/<repo>/blob/<commit>/<path> pinned to a full commit,
    cached under ckpts/github/<owner>/<repo>/<commit>/<path>."""
    from methods.core.weights import ROOT
    local = Path(str(source)).expanduser()
    local = local if local.is_absolute() else ROOT / local
    if local.is_file():
        return local
    url = urlparse(str(source))
    parts = url.path.strip("/").split("/")
    if (url.scheme != "https" or url.netloc != "github.com" or len(parts) < 5 or parts[2] != "blob"
            or not re.fullmatch(r"[0-9a-f]{40}", parts[3])):
        raise ValueError(f"Expected a local file or a commit-pinned github.com blob URL: {source}")
    owner, repo, commit, name = parts[0], parts[1], parts[3], "/".join(parts[4:])
    target = ROOT / "ckpts" / "github" / owner / repo / commit / name
    if target.is_file():
        return target
    if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
        raise FileNotFoundError(f"{target} is not cached; run prepare first")
    import urllib.request
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.{os.getpid()}.part")
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{name}"
    try:
        with urllib.request.urlopen(raw, timeout=300) as response, part.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target


def resolve_weights(cfg, lifecycle):
    """prepare fetches every file infer will need; infer then only reads them offline.

    train needs no VAE and no weights.checkpoint: the trained head is its output."""
    from methods.core.weights import directory, checkpoint, require, validate_shards
    if lifecycle == "infer" and uses_head(cfg) and not cfg.weights.checkpoint:
        raise ValueError("H3 router inference requires weights.checkpoint or forced_exit_step")
    decode = cfg.decode and lifecycle != "train"
    parts = ["transformer"] + (["vae"] if decode and cfg.decoder == "vae" else [])
    if lifecycle == "prepare":
        parts = list(dict.fromkeys([*PREPARE_PARTS, *parts]))
    base = directory(cfg.weights.base, patterns=[f"{part}/*" for part in parts] + ["*.json"])
    for part in parts:
        require(base / part)
        if part in MODEL_PARTS:
            require(base / part / "config.json")
            validate_shards(base / part)
    stage2 = directory(cfg.weights.stage2, patterns=["ema.pt", "*.json"])
    for name in STAGE2_FILES:
        require(stage2 / name)
    cfg.base_path, cfg.stage2_path = str(base), str(stage2)
    cfg.taeh3_path = str(github_file(cfg.weights.taeh3)) if decode and cfg.decoder == "taeh3" else None
    cfg.trained_path = (str(checkpoint(cfg.weights.checkpoint, filename="MiniMax-H3/model.pt"))
                        if lifecycle != "train" and uses_head(cfg) and cfg.weights.checkpoint else None)
    if lifecycle == "train":
        cfg.resume_path = str(checkpoint(cfg.train.resume)) if cfg.train.resume else None
        cfg.init_head_path = str(checkpoint(cfg.train.init_head)) if cfg.train.init_head else None
    return cfg
