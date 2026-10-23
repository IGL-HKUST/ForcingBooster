"""DL3DV windows for SolarWM: the seeded scene manifest and 158-frame first-image conditions.

Scene validity, splits and image geometry reuse LingBot's generic DL3DV reader
(``methods.LingBotWorld.shared.data.dl3dv``), imported lazily: it pulls torch and scipy.

python -m methods.H3.SolarWM.dl3dv build --root R --train N --validation M --out manifest.json
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np

SCHEMA = "forcingbooster.h3-data.v1"
DEFAULT_ROOT = "dataset/dl3dv10k"
SPLIT_INTERVAL = 20
CANONICAL_FRAMES = 115  # LingBot's scene validity
NUM_FRAMES = 158        # H3 eligibility: one window of 158 consecutive source frames
WIDTH, HEIGHT = 1344, 768
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
PREPROCESSING = dict(undistort="same_K", crop="center", resize="bicubic", height=HEIGHT, width=WIDTH,
                     fps=24, camera="absolute_opengl_to_opencv_relative_first_w2c",
                     camera_sampling="solar_1_4_4_4_4_repeat_codec_tail", anchor_seed=42,
                     visual_vae_parameters="fp32_native", anchor_rounding="float16_then_normalize")


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _lingbot():
    from methods.LingBotWorld.shared.data import dl3dv
    return dl3dv


def window_start(seed, split, sample_id, frame_count):
    """Train windows start at a seeded position; validation windows are centred."""
    if frame_count < NUM_FRAMES:
        raise ValueError(f"{sample_id} has {frame_count} frames, need {NUM_FRAMES}")
    if split == "validation":
        return (frame_count - NUM_FRAMES) // 2
    return int(_digest([seed, sample_id])[:16], 16) % (frame_count - NUM_FRAMES + 1)


def _frames(data):
    frames = sorted(data["frames"], key=_lingbot()._frame_number)
    numbers = [_lingbot()._frame_number(f) for f in frames]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("transforms.json frames are not consecutive")
    return frames


def _caption(data):
    for key in ("long_caption", "medium_caption", "short_caption"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("caption.json contains no usable caption")


def _camera(data, window, actual_size, crop_box):
    """Sampled first-frame-relative OpenCV W2C and normalized K of a 158-frame window."""
    import torch
    from methods.LingBotWorld.shared.utils.camera import transform_pixel_intrinsics
    from .backbone.camera import first_frame_relative_w2c, validate_normalized_intrinsics
    from .backbone.full_length import full_length_geometry
    indices = list(full_length_geometry(len(window)).camera_indices)
    c2w = np.asarray([f["transform_matrix"] for f in window], dtype=np.float64) @ OPENGL_TO_OPENCV
    views = first_frame_relative_w2c(c2w)[indices]
    pixel, source_size = _lingbot()._pixel_intrinsics(data, window[0])
    fx, fy, cx, cy = transform_pixel_intrinsics(
        pixel, source_size=source_size, actual_size=tuple(actual_size), crop_box=tuple(crop_box),
        target_size=(WIDTH, HEIGHT)).astype(np.float64) / np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    validate_normalized_intrinsics(K[None])
    return dict(viewmats=torch.from_numpy(views.astype(np.float32)),
                K=torch.from_numpy(K).repeat(len(indices), 1, 1))


def make_sample(root, entry, seed):
    """One manifest sample: LingBot's scene entry, its H3 window and the raw-file identity."""
    from PIL import Image
    from .backbone.full_length import full_length_geometry
    root = Path(root)
    geometry = full_length_geometry(NUM_FRAMES)
    sample_id = f"{entry['bucket']}/{entry['sequence_id']}"
    start = window_start(seed, entry["split"], sample_id, int(entry["frame_count"]))
    transform = (root / entry["transform_path"]).read_bytes()
    caption = (root / entry["caption_path"]).read_bytes()
    data = json.loads(transform)
    frames = _frames(data)
    if len(frames) != entry["frame_count"]:
        raise ValueError(f"{sample_id}: transforms.json changed since enumeration")
    window = frames[start:start + NUM_FRAMES]
    first_image = f"{entry['image_dir']}/{Path(str(window[0]['file_path'])).name}"
    image = (root / first_image).read_bytes()
    with Image.open(io.BytesIO(image)) as handle:
        actual_size = list(handle.size)
    crop_box = list(_lingbot()._center_crop_box(*actual_size, WIDTH, HEIGHT))
    _lingbot()._distortion_coefficients(data, window[0])
    _camera(data, window, actual_size, crop_box)  # fail here, not in the middle of a GPU prepare
    return dict(**entry, sample_id=sample_id, start=start, num_frames=NUM_FRAMES,
                decode_latents=geometry.decode_latents, rollout_latents=geometry.rollout_latents,
                preprocessing=dict(PREPROCESSING),
                raw_identity=dict(transform_sha256=_sha256(transform), caption_sha256=_sha256(caption),
                                  first_image_sha256=_sha256(image), caption=_caption(json.loads(caption)),
                                  first_image=first_image, actual_size=actual_size, crop_box=crop_box))


def enumerate_scenes(root):
    """LingBot's canonical enumeration (115-frame validity, every 20th valid scene validates).

    Reads every scene's transforms.json: run it on a pod, not on the jump host."""
    return _lingbot().enumerate_dl3dv_scenes(Path(root), split_interval=SPLIT_INTERVAL,
                                             required_source_frames=CANONICAL_FRAMES)


def eligible_entries(scenes, split, seed=42):
    """Entries of one split with a full H3 window, in seeded selection order."""
    entries = [e for e in scenes["entries"] if e["split"] == split and e["frame_count"] >= NUM_FRAMES]
    return sorted(entries, key=lambda e: _digest([seed, split, f"{e['bucket']}/{e['sequence_id']}"]))


def build_manifest(root, *, train, validation, seed=42, scenes=None):
    """The H3 DL3DV manifest: the first ``train``/``validation`` eligible scenes per split.

    Selection is a per-split sha256 order, so a larger manifest extends a smaller one.
    Pass ``scenes`` (from ``enumerate_scenes``) to reuse one enumeration."""
    root = Path(root).expanduser().resolve()
    scenes = enumerate_scenes(root) if scenes is None else scenes
    limits = dict(train=int(train), validation=int(validation))
    samples = []
    for split, limit in limits.items():
        entries = eligible_entries(scenes, split, seed)
        if limit < 0 or len(entries) < limit:
            raise ValueError(f"{split}: requested {limit} samples, {len(entries)} eligible")
        samples += [make_sample(root, entry, seed) for entry in entries[:limit]]
    return dict(schema=SCHEMA, seed=seed, canonical_eligibility=CANONICAL_FRAMES,
                h3_eligibility=NUM_FRAMES, split_interval=SPLIT_INTERVAL,
                canonical_statistics=scenes["statistics"], samples=samples,
                selection=dict(method="sha256_seed_split_scene", limits=limits))


def dump_manifest(manifest):
    # Keep this formatting: manifests written earlier must stay byte-identical.
    return json.dumps(manifest, indent=2, ensure_ascii=False)


def write_manifest(manifest, out):
    """Write a new manifest; never overwrite one (condition keys depend on its samples)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(dump_manifest(manifest))
    return out


def read_dl3dv(sample, root):
    """First image (1344x768 PIL), caption and SolarWM camera of one DL3DV manifest sample.

    The image is undistorted with the same pixel K (cv2), centre-cropped and resized
    bicubically, as LingBot's DL3DV reader does; raw files must match ``raw_identity``."""
    from PIL import Image
    from methods.LingBotWorld.shared.utils.camera import (transform_pixel_intrinsics,
                                                          undistort_image_and_intrinsics)
    lingbot = _lingbot()
    root, raw, name = Path(root), sample["raw_identity"], sample["sample_id"]
    files = dict(transform=sample["transform_path"], caption=sample["caption_path"], first_image=raw["first_image"])
    blobs = {k: (root / v).read_bytes() for k, v in files.items()}
    for key, blob in blobs.items():
        if _sha256(blob) != raw[f"{key}_sha256"]:
            raise ValueError(f"{name}: {files[key]} differs from the manifest's raw_identity")
    data = json.loads(blobs["transform"])
    frames = _frames(data)
    if "frame_count" in sample and len(frames) != sample["frame_count"]:
        raise ValueError(f"{name}: transforms.json has {len(frames)} frames, manifest {sample['frame_count']}")
    window = frames[sample["start"]:sample["start"] + sample["num_frames"]]
    if len(window) != sample["num_frames"]:
        raise ValueError(f"{name}: window {sample['start']}+{sample['num_frames']} exceeds {len(frames)} frames")
    if Path(str(window[0]["file_path"])).name != Path(raw["first_image"]).name:
        raise ValueError(f"{name}: first window frame is not {raw['first_image']}")
    caption = _caption(json.loads(blobs["caption"]))
    if caption != raw["caption"]:
        raise ValueError(f"{name}: caption differs from the manifest's raw_identity")

    with Image.open(io.BytesIO(blobs["first_image"])) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    crop_box = lingbot._center_crop_box(width, height, WIDTH, HEIGHT)
    if [width, height] != list(raw["actual_size"]) or list(crop_box) != list(raw["crop_box"]):
        raise ValueError(f"{name}: image size/crop differ from the manifest's raw_identity")
    pixel, source_size = lingbot._pixel_intrinsics(data, window[0])
    actual = transform_pixel_intrinsics(pixel, source_size=source_size, actual_size=(width, height),
                                        crop_box=(0, 0, width, height), target_size=(width, height))
    rgb, same = undistort_image_and_intrinsics(rgb, actual, lingbot._distortion_coefficients(data, window[0]))
    if not np.array_equal(same, actual):
        raise AssertionError("same_K undistortion changed the pixel intrinsics")
    image = Image.fromarray(rgb).crop(crop_box).resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
    return image, caption, _camera(data, window, (width, height), crop_box)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SolarWM DL3DV manifest (CPU only)")
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", default=DEFAULT_ROOT)
    build.add_argument("--train", type=int, required=True)
    build.add_argument("--validation", type=int, required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    scenes = enumerate_scenes(Path(args.root).expanduser().resolve())
    manifest = build_manifest(args.root, train=args.train, validation=args.validation,
                              seed=args.seed, scenes=scenes)
    write_manifest(manifest, out)
    stats = scenes["statistics"]
    print(json.dumps(dict(out=str(out), candidates=stats["candidate_count"], valid=stats["valid_count"],
                          valid_splits=stats["counts"],
                          eligible={s: len(eligible_entries(scenes, s, args.seed)) for s in ("train", "validation")},
                          selected={s: sum(x["split"] == s for x in manifest["samples"])
                                    for s in ("train", "validation")}), indent=2))


if __name__ == "__main__":
    main()
