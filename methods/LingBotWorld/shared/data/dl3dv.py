"""Deterministic DL3DV windows for LingBot-World v2 camera conditioning."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from ..utils.camera import (
    interpolate_camera_poses,
    opengl_to_opencv_c2w,
    relative_adjacent_poses,
    transform_pixel_intrinsics,
    undistort_image_and_intrinsics,
)
from .contracts import (
    VAE_TEMPORAL_STRIDE,
    LingBotV2I2VSample,
    LingBotV2WindowSpec,
    WINDOW_PROFILES,
)


DISTORTION_POLICY = "opencv_same_k"
_FRAME_NUMBER_RE = re.compile(r"(\d+)$")
_DISTORTION_KEY_RE = re.compile(r"^[kp]\d+$")
_SUPPORTED_DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3")
_MISSING = object()


def _frame_number(frame: Mapping[str, Any]) -> int:
    stem = Path(str(frame.get("file_path", ""))).stem
    match = _FRAME_NUMBER_RE.search(stem)
    if match is None:
        raise ValueError(f"frame path has no numeric suffix: {frame.get('file_path')!r}")
    return int(match.group(1))


def _frame_or_global(
    data: Mapping[str, Any],
    frame: Mapping[str, Any],
    key: str,
    default: Any = _MISSING,
) -> Any:
    if key in frame:
        return frame[key]
    if key in data:
        return data[key]
    if default is _MISSING:
        raise KeyError(f"missing Nerfstudio camera field {key!r}")
    return default


def _pixel_intrinsics(
    data: Mapping[str, Any], frame: Mapping[str, Any]
) -> tuple[np.ndarray, tuple[int, int]]:
    values = np.asarray(
        [
            _frame_or_global(data, frame, "fl_x"),
            _frame_or_global(data, frame, "fl_y"),
            _frame_or_global(data, frame, "cx"),
            _frame_or_global(data, frame, "cy"),
        ],
        dtype=np.float64,
    )
    width = int(_frame_or_global(data, frame, "w"))
    height = int(_frame_or_global(data, frame, "h"))
    if not np.isfinite(values).all() or np.any(values[:2] <= 0):
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    if width <= 0 or height <= 0:
        raise ValueError("camera metadata width and height must be positive")
    return values.astype(np.float32), (width, height)


def _distortion_coefficients(
    data: Mapping[str, Any], frame: Mapping[str, Any]
) -> np.ndarray:
    camera_model = str(_frame_or_global(data, frame, "camera_model", "PINHOLE")).upper()
    if camera_model not in {"OPENCV", "PINHOLE"}:
        raise ValueError(f"unsupported DL3DV camera_model {camera_model!r}")
    for container in (data, frame):
        for key, raw_value in container.items():
            if (
                _DISTORTION_KEY_RE.fullmatch(str(key))
                and key not in _SUPPORTED_DISTORTION_KEYS
                and abs(float(raw_value)) > 0.0
            ):
                raise ValueError(f"unsupported non-zero distortion field {key!r}")
    coefficients = np.asarray(
        [
            _frame_or_global(data, frame, "k1", 0.0),
            _frame_or_global(data, frame, "k2", 0.0),
            _frame_or_global(data, frame, "p1", 0.0),
            _frame_or_global(data, frame, "p2", 0.0),
            _frame_or_global(data, frame, "k3", 0.0),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(coefficients).all():
        raise ValueError("camera distortion coefficients contain non-finite values")
    if camera_model == "PINHOLE" and not np.allclose(coefficients, 0.0):
        raise ValueError("PINHOLE camera metadata cannot contain non-zero distortion")
    return coefficients.astype(np.float32)


def _center_crop_box(
    width: int, height: int, target_width: int, target_height: int
) -> tuple[int, int, int, int]:
    target_ratio = target_width / float(target_height)
    image_ratio = width / float(height)
    if image_ratio > target_ratio:
        crop_height = height
        crop_width = int(round(height * target_ratio))
        left = (width - crop_width) // 2
        top = 0
    else:
        crop_width = width
        crop_height = int(round(width / target_ratio))
        left = 0
        top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def _resolve_under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"scene path escapes the dataset root: {relative!r}") from exc
    return candidate


def _actual_image_path(image_dir: Path, frame: Mapping[str, Any]) -> Path:
    path = image_dir / Path(str(frame["file_path"])).name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_transform(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    frames_raw = data.get("frames")
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ValueError(f"{path} has no camera frames")
    frames = sorted(frames_raw, key=_frame_number)
    frame_numbers = np.asarray([_frame_number(frame) for frame in frames], dtype=np.int64)
    if len(frame_numbers) > 1 and not np.all(np.diff(frame_numbers) == 1):
        raise ValueError(f"{path} contains duplicate or missing source frame numbers")
    return data, frames


def _validate_scene(
    *,
    transform_path: Path,
    image_dir: Path,
    caption_path: Path,
    required_source_frames: int,
) -> int:
    data, frames = _load_transform(transform_path)
    if len(frames) < required_source_frames:
        raise ValueError(
            f"{transform_path} has {len(frames)} frames, need {required_source_frames}"
        )
    poses = np.asarray([frame.get("transform_matrix") for frame in frames], dtype=np.float64)
    opengl_to_opencv_c2w(poses)
    for frame in frames:
        _pixel_intrinsics(data, frame)
        _distortion_coefficients(data, frame)
    # The transform list is already checked for consecutive numeric names.  A
    # full per-image stat over DL3DV makes prepare prohibitively slow on PFS;
    # representative boundaries catch directory/layout errors and __getitem__
    # remains fail-fast for the selected first frame.
    representative_indices = {0, required_source_frames - 1, len(frames) - 1}
    for frame_index in sorted(representative_indices):
        _actual_image_path(image_dir, frames[frame_index])
    if not caption_path.is_file():
        raise FileNotFoundError(caption_path)
    with caption_path.open(encoding="utf-8") as handle:
        caption = json.load(handle)
    if not any(
        isinstance(caption.get(key), str) and caption[key].strip()
        for key in ("long_caption", "medium_caption", "short_caption")
    ):
        raise ValueError(f"{caption_path} contains no usable caption")
    return len(frames)


def _filter_reason(exc: Exception) -> str:
    message = str(exc).lower()
    if "frames, need" in message:
        return "insufficient_frames"
    if "duplicate or missing source frame numbers" in message:
        return "missing_source_frames"
    if isinstance(exc, FileNotFoundError):
        return "missing_image_file"
    if "caption" in message:
        return "invalid_caption"
    if any(token in message for token in ("camera", "intrinsic", "rotation", "pose")):
        return "invalid_camera"
    return "invalid_scene"


def _caption_bucket(bucket: str) -> str:
    return bucket.split("-", 1)[0]


def enumerate_dl3dv_scenes(
    data_root: str | Path,
    *,
    split_interval: int = 20,
    required_source_frames: int = 115,
    max_scenes: int | None = None,
) -> dict[str, Any]:
    """Enumerate valid scenes in bucket/scene order and assign in-memory splits."""
    root = Path(data_root).expanduser().resolve()
    images_root = root / "images+poses"
    captions_root = root / "caption_json_files"
    if split_interval <= 1:
        raise ValueError("split_interval must be greater than one")
    if required_source_frames < WINDOW_PROFILES["train"].source_frames:
        raise ValueError(
            "required_source_frames must cover the 115-frame training profile"
        )
    if not images_root.is_dir() or not captions_root.is_dir():
        raise FileNotFoundError("DL3DV root must contain images+poses and caption_json_files")

    scenes: list[tuple[str, Path]] = []
    for bucket_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        for scene_dir in sorted(path for path in bucket_dir.iterdir() if path.is_dir()):
            scenes.append((bucket_dir.name, scene_dir))
    if max_scenes is not None:
        if max_scenes <= 0:
            raise ValueError("max_scenes must be positive")
        scenes = scenes[:max_scenes]

    valid_entries: list[dict[str, Any]] = []
    filtered_counts: dict[str, int] = {}
    filtered_examples: dict[str, list[dict[str, str]]] = {}
    for bucket, scene_dir in scenes:
        sequence_id = scene_dir.name
        transform_path = scene_dir / "transforms.json"
        image_dir = scene_dir / "images_4"
        caption_path = captions_root / _caption_bucket(bucket) / sequence_id / "caption.json"
        reason = None
        exc: Exception | None = None
        if not transform_path.is_file():
            reason = "missing_transform"
            exc = FileNotFoundError(transform_path)
        elif not image_dir.is_dir():
            reason = "missing_image_dir"
            exc = FileNotFoundError(image_dir)
        elif not caption_path.is_file():
            reason = "missing_caption"
            exc = FileNotFoundError(caption_path)
        else:
            try:
                frame_count = _validate_scene(
                    transform_path=transform_path,
                    image_dir=image_dir,
                    caption_path=caption_path,
                    required_source_frames=required_source_frames,
                )
            except Exception as validation_error:
                reason = _filter_reason(validation_error)
                exc = validation_error
        if reason is not None:
            assert exc is not None
            filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
            examples = filtered_examples.setdefault(reason, [])
            if len(examples) < 8:
                error_message = str(exc).replace(str(root), "<DATA_ROOT>")
                examples.append(
                    {
                        "scene": f"{bucket}/{sequence_id}",
                        "error": error_message[:500],
                    }
                )
            continue
        valid_entries.append(
            {
                "bucket": bucket,
                "caption_bucket": _caption_bucket(bucket),
                "sequence_id": sequence_id,
                "transform_path": transform_path.relative_to(root).as_posix(),
                "image_dir": image_dir.relative_to(root).as_posix(),
                "caption_path": caption_path.relative_to(root).as_posix(),
                "frame_count": frame_count,
            }
        )

    if not valid_entries:
        raise RuntimeError(f"no DL3DV scenes found under {root}")
    entries = []
    for valid_index, entry in enumerate(valid_entries):
        entries.append(
            {
                "index": valid_index,
                "split": (
                    "validation"
                    if (valid_index + 1) % split_interval == 0
                    else "train"
                ),
                **entry,
            }
        )
    counts = {
        split: sum(entry["split"] == split for entry in entries)
        for split in ("train", "validation")
    }
    return {
        "entries": entries,
        "statistics": {
            "candidate_count": len(scenes),
            "valid_count": len(entries),
            "filtered_counts": filtered_counts,
            "filtered_examples": filtered_examples,
            "counts": counts,
        },
    }


def _load_caption(path: Path, caption_key: str) -> str:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get(caption_key)
    if not isinstance(value, str) or not value.strip():
        value = data.get("long_caption") or data.get("medium_caption") or data.get(
            "short_caption"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"no usable caption in {path}")
    return value.strip()


class LingBotV2DL3DVDataset(Dataset):
    """Return rank-zero LingBot raw conditions without loading T5 or VAE."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        entries: Sequence[Mapping[str, Any]] | None = None,
        split_interval: int = 20,
        required_source_frames: int = 115,
        max_scenes: int | None = None,
        split: str = "train",
        profile: str = "train",
        target_height: int = 480,
        target_width: int = 832,
        caption_key: str = "long_caption",
        random_sample: bool = True,
        seed: int = 42,
        max_samples: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if profile not in WINDOW_PROFILES:
            raise ValueError(f"unknown window profile {profile!r}")
        profile_window = WINDOW_PROFILES[profile]
        self.window: LingBotV2WindowSpec = (
            profile_window
            if chunk_size is None
            else replace(profile_window, chunk_size=int(chunk_size))
        )
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.data_root = Path(data_root).expanduser().resolve()
        if entries is None:
            entries = enumerate_dl3dv_scenes(
                self.data_root,
                split_interval=split_interval,
                required_source_frames=required_source_frames,
                max_scenes=max_scenes,
            )["entries"]
        self.entries = [dict(entry) for entry in entries if entry["split"] == split]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.entries = self.entries[:max_samples]
        if not self.entries:
            raise RuntimeError(f"DL3DV contains no {split} entries")
        self.split = split
        self.profile = profile
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError("target image dimensions must be positive")
        self.caption_key = str(caption_key)
        self.random_sample = bool(random_sample)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _window_start(self, item_index: int, frame_count: int) -> int:
        maximum = frame_count - self.window.source_frames
        if maximum < 0:
            raise ValueError(
                f"scene has {frame_count} frames, need {self.window.source_frames}"
            )
        if maximum == 0:
            return 0
        if not self.random_sample:
            return maximum // 2
        payload = f"{self.seed}:{self.epoch}:{item_index}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return value % (maximum + 1)

    def __getitem__(self, item_index: int) -> LingBotV2I2VSample:
        entry = self.entries[item_index]
        transform_path = _resolve_under(self.data_root, entry["transform_path"])
        image_dir = _resolve_under(self.data_root, entry["image_dir"])
        caption_path = _resolve_under(self.data_root, entry["caption_path"])
        data, frames = _load_transform(transform_path)
        start = self._window_start(item_index, len(frames))
        source_frames = frames[start : start + self.window.source_frames]

        source_c2ws = np.asarray(
            [frame["transform_matrix"] for frame in source_frames], dtype=np.float64
        )
        source_c2ws = opengl_to_opencv_c2w(source_c2ws)
        source_times = np.arange(self.window.source_frames, dtype=np.float64)
        target_times = self.window.target_source_times().numpy()
        interpolated_c2ws = interpolate_camera_poses(
            source_c2ws, source_times, target_times
        )
        latent_c2ws = interpolated_c2ws[::VAE_TEMPORAL_STRIDE]
        if len(latent_c2ws) != self.window.latent_frames:
            raise AssertionError("latent camera sampling violated the window contract")
        camera_c2ws, translation_scale, static_camera = relative_adjacent_poses(
            latent_c2ws
        )

        first_image_path = _actual_image_path(image_dir, source_frames[0])
        with Image.open(first_image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        actual_height, actual_width = rgb.shape[:2]
        crop_box = _center_crop_box(
            actual_width,
            actual_height,
            self.target_width,
            self.target_height,
        )

        source_intrinsics = []
        source_sizes = []
        for frame in source_frames:
            values, source_size = _pixel_intrinsics(data, frame)
            source_intrinsics.append(values)
            source_sizes.append(source_size)
        if len(set(source_sizes)) != 1:
            raise ValueError("per-frame source image dimensions are not supported")
        transformed_intrinsics = transform_pixel_intrinsics(
            np.stack(source_intrinsics),
            source_size=source_sizes[0],
            actual_size=(actual_width, actual_height),
            crop_box=crop_box,
            target_size=(self.target_width, self.target_height),
        )
        target_intrinsics = np.stack(
            [
                np.interp(target_times, source_times, transformed_intrinsics[:, component])
                for component in range(4)
            ],
            axis=-1,
        ).astype(np.float32)
        latent_intrinsics = target_intrinsics[::VAE_TEMPORAL_STRIDE]

        first_raw_intrinsics, source_size = _pixel_intrinsics(data, source_frames[0])
        first_actual_intrinsics = transform_pixel_intrinsics(
            first_raw_intrinsics,
            source_size=source_size,
            actual_size=(actual_width, actual_height),
            crop_box=(0, 0, actual_width, actual_height),
            target_size=(actual_width, actual_height),
        )
        distortion = _distortion_coefficients(data, source_frames[0])
        rgb, undistorted_intrinsics = undistort_image_and_intrinsics(
            rgb, first_actual_intrinsics, distortion
        )
        if not np.allclose(
            first_actual_intrinsics, undistorted_intrinsics, rtol=0.0, atol=0.0
        ):
            raise AssertionError("opencv_same_k undistortion changed camera intrinsics")
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        processed = Image.fromarray(rgb).crop(crop_box).resize(
            (self.target_width, self.target_height), resampling
        )
        first_frame = torch.from_numpy(np.array(processed, dtype=np.uint8, copy=True))
        first_frame = first_frame.permute(2, 0, 1).contiguous().float() / 127.5 - 1.0

        source_numbers = [_frame_number(frame) for frame in source_frames]
        metadata = {
            "scene_index": int(entry["index"]),
            "split": self.split,
            "profile": self.profile,
            "bucket": entry["bucket"],
            "sequence_id": entry["sequence_id"],
            "source_frame_start": source_numbers[0],
            "source_frame_end": source_numbers[-1],
            "source_frame_count": self.window.source_frames,
            "source_fps": self.window.source_fps,
            "target_fps": self.window.target_fps,
            "video_frame_count": self.window.video_frames,
            "latent_frame_count": self.window.latent_frames,
            "chunk_count": self.window.chunks,
            "chunk_frames": self.window.chunk_size,
            "camera_pose_convention": "opencv_c2w_adjacent_clip_max",
            "translation_scale": translation_scale,
            "static_camera": static_camera,
            "intrinsics_convention": "pixel_fx_fy_cx_cy_after_opencv_same_k_crop_resize",
            "distortion_policy": DISTORTION_POLICY,
            "i2v_condition": {
                "mask_channels": 4,
                "vae_channels": 16,
                "output_channels": 20,
                "builder": "assemble_i2v_condition",
            },
        }
        return {
            "prompt": _load_caption(caption_path, self.caption_key),
            "first_frame": first_frame,
            "camera_c2ws": torch.from_numpy(camera_c2ws),
            "camera_intrinsics": torch.from_numpy(latent_intrinsics),
            "sample_metadata": metadata,
        }


def make_lingbot_v2_distributed_sampler(
    dataset: Dataset,
    *,
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
) -> DistributedSampler:
    """Build one deterministic, non-padding sampler shard for a training rank."""

    rank = int(rank)
    world_size = int(world_size)
    epoch = int(epoch)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size={world_size}")
    if len(dataset) < world_size:
        raise ValueError(
            "LingBot distributed training requires at least one sample per rank: "
            f"dataset_size={len(dataset)}, world_size={world_size}"
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(seed),
        drop_last=True,
    )
    sampler.set_epoch(epoch)
    return sampler


def lingbot_v2_collate(samples: Sequence[LingBotV2I2VSample]) -> dict[str, Any]:
    """Collate raw samples for rank-zero T5/VAE conditioning and broadcast."""
    if not samples:
        raise ValueError("cannot collate an empty LingBot batch")
    return {
        "prompt": [sample["prompt"] for sample in samples],
        "first_frame": torch.stack([sample["first_frame"] for sample in samples]),
        "camera_c2ws": torch.stack([sample["camera_c2ws"] for sample in samples]),
        "camera_intrinsics": torch.stack(
            [sample["camera_intrinsics"] for sample in samples]
        ),
        "sample_metadata": [sample["sample_metadata"] for sample in samples],
    }


LingBotDL3DVDataset = LingBotV2DL3DVDataset
collate_lingbot_samples = lingbot_v2_collate


__all__ = [
    "DISTORTION_POLICY",
    "LingBotDL3DVDataset",
    "LingBotV2DL3DVDataset",
    "enumerate_dl3dv_scenes",
    "collate_lingbot_samples",
    "lingbot_v2_collate",
    "make_lingbot_v2_distributed_sampler",
]
