"""Resumable router bundle: FP32 head, AdamW state, rank RNG, data cursor and contract."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

SCHEMA = "forcingbooster.h3-router.v1"
CONTAINS = ["FP32 router head", "AdamW optimizer", "rank RNG", "data cursor and order",
            "training contract", "frozen parent fingerprints"]
KEYS = ("schema", "head", "optimizer", "identity", "data_identity", "contract", "head_dim",
        "updates", "attempts", "cursor", "order", "epoch", "rank_rng", "learning_rate",
        "exit_threshold", "experiment")


def epoch_order(seed, epoch, n):
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace(target, write):
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    write(tmp)
    os.replace(tmp, target)


def _save(payload, filename):
    with open(filename, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(root, payload, *, step, resolved_config=None):
    """Rank zero only. Bundle, manifest and config first; ``latest`` moves last."""
    root = Path(root)
    name = f"checkpoint_step_{int(step):06d}"
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / "model.pt"
    _replace(model, lambda tmp: _save(payload, tmp))
    manifest = dict(schema=SCHEMA, step=int(step), checkpoint=model.name, sha256=sha256(model),
                    contains=CONTAINS, updates=int(payload["updates"]), attempts=int(payload["attempts"]))
    _replace(folder / "manifest.json", lambda tmp: tmp.write_text(json.dumps(manifest, indent=2) + "\n"))
    if resolved_config is not None:
        _replace(folder / "resolved_config.yaml", lambda tmp: tmp.write_text(resolved_config))
    _replace(root / "latest", lambda tmp: tmp.symlink_to(name))
    return model


def load_bundle(path):
    """A model.pt, a checkpoint directory, or ``latest`` pointing at one."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_resume(bundle, *, contract, head_dim, learning_rate, exit_threshold, seed, num_train):
    """Refuse optimizer state that this run's recipe and data order would not have produced."""
    def check(ok, message):
        if not ok:
            raise ValueError(f"Cannot resume H3 router bundle: {message}")
    check(isinstance(bundle, dict), f"expected a dict, got {type(bundle).__name__}")
    check(bundle.get("schema") == SCHEMA, f"schema {bundle.get('schema')!r} is not {SCHEMA!r}")
    missing = [key for key in KEYS if key not in bundle]
    check(not missing, f"missing keys {missing}")
    check(bundle["contract"] == contract,
          f"training contract {bundle['contract']} differs from {contract}; "
          "load the head as initialization instead of resuming")
    check(int(bundle["head_dim"]) == int(head_dim), f"head_dim {bundle['head_dim']} differs from {head_dim}")
    for key, value in dict(learning_rate=learning_rate, exit_threshold=exit_threshold).items():
        check(float(bundle[key]) == float(value), f"{key} {bundle[key]!r} differs from {value!r}")
    updates, attempts = int(bundle["updates"]), int(bundle["attempts"])
    check(0 <= updates <= attempts, f"updates {updates} and attempts {attempts} violate 0 <= updates <= attempts")
    state = bundle["optimizer"]["state"]
    check(state or not updates, f"optimizer state is empty after {updates} updates")
    steps = sorted({float(entry["step"]) for entry in state.values()})
    check(steps in ([], [float(updates)]), f"optimizer steps {steps} differ from updates {updates}")
    epoch, cursor = int(bundle["epoch"]), int(bundle["cursor"])
    check(list(bundle["order"]) == epoch_order(seed, epoch, num_train),
          f"data order is not epoch {epoch} of seed {seed} over {num_train} training samples")
    check(0 <= cursor <= num_train, f"cursor {cursor} outside [0, {num_train}]")
    bad = [key for key, value in bundle["head"].items() if not torch.isfinite(value).all()]
    check(not bad, f"non-finite head tensors {bad}")
