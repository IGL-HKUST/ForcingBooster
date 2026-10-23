"""LingBot example files, prepared manifests and condition directories."""
import hashlib
import json
from pathlib import Path
from .config import path

LINGBOT_FILES = ("image.jpg", "prompt.txt", "poses.npy", "intrinsics.npy")


def condition_key(sample):
    # Preserve existing manifest-to-directory keys without retaining receipts.
    return hashlib.sha256(json.dumps(sample, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def condition_dir(cfg, sample):
    return path(cfg.conditions_dir) / condition_key(sample)


def sample_names(sample):
    """Every name a sample answers to; the first one names its output directory."""
    names = []
    if "display_name" in sample:
        names.append(str(sample["display_name"]))
    if "index" in sample and "sequence_id" in sample:
        names.append(f"dl3dv_{int(sample['index']):05d}")
    if "image" in sample.get("source_files", {}):
        names.append(Path(sample["source_files"]["image"]).parent.name)
    if "sample_id" in sample:
        names += [str(sample["sample_id"]).replace("/", "_"), str(sample["sample_id"])]
    if not names:
        raise ValueError(f"H3 sample has no usable name: {sorted(sample)}")
    return names


def sample_name(sample):
    return sample_names(sample)[0]


def select_samples(samples, cfg, input_kind=None):
    if input_kind is not None:
        samples = [s for s in samples if s.get("input_kind") == input_kind]
    if cfg.get("split"):
        samples = [s for s in samples if s.get("split") == cfg.split]
    if cfg.sample_ids:
        wanted = {str(x) for x in cfg.sample_ids}
        samples = [s for s in samples if wanted & set(sample_names(s))]
        found = {name for s in samples for name in sample_names(s)}
        if wanted - found:
            raise ValueError(f"Unknown H3 samples: {sorted(wanted - found)}")
    if not samples:
        raise ValueError("No H3 samples selected")
    return sorted(samples, key=sample_name)


def read_manifest(cfg):
    filename = path(cfg.manifest) if cfg.manifest else path(cfg.conditions_dir) / "manifest.json"
    return json.loads(filename.read_text())["samples"]


def lingbot_folders(root):
    """Example folders holding an image, prompt, camera poses and intrinsics."""
    folders = []
    for folder in sorted(path(root).iterdir()):
        if not folder.is_dir():
            continue
        files = {name.split(".")[0]: folder / name for name in LINGBOT_FILES}
        for filename in files.values():
            if not filename.is_file():
                raise FileNotFoundError(filename)
        folders.append((folder, files))
    return folders


def read_lingbot(sample):
    """Raw image, caption, poses and intrinsics of a LingBot example sample."""
    import numpy as np
    from PIL import Image
    files = {k: path(v) for k, v in sample["source_files"].items()}
    poses = np.load(files["poses"], allow_pickle=False)
    intrinsics = np.load(files["intrinsics"], allow_pickle=False)
    image = Image.open(files["image"]).convert("RGB")
    caption = (files["image"].parent / "prompt.txt").read_text().strip()
    return image, caption, poses, intrinsics
