"""Camera geometry used by the LingBot-World v2 data and model adapters.

The upstream project calls its six-channel camera representation "Plucker",
but the released implementation concatenates a ray origin and a unit ray
direction.  This module intentionally preserves that released behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from numpy.typing import NDArray


_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _as_pose_array(c2ws: np.ndarray) -> np.ndarray:
    poses = np.asarray(c2ws, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"camera poses must have shape (F, 4, 4), got {poses.shape}")
    if len(poses) == 0:
        raise ValueError("camera poses must not be empty")
    if not np.isfinite(poses).all():
        raise ValueError("camera poses contain non-finite values")
    expected_bottom = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        poses[:, 3, :].shape,
    )
    if not np.allclose(poses[:, 3, :], expected_bottom, rtol=0.0, atol=1e-5):
        raise ValueError("camera poses must be homogeneous rigid transforms")
    rotations = poses[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    orthogonality = rotations.transpose(0, 2, 1) @ rotations
    if not np.allclose(orthogonality, identity[None], rtol=0.0, atol=2e-3):
        raise ValueError("camera rotations are not orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, rtol=0.0, atol=2e-3):
        raise ValueError("camera rotations must have determinant +1")
    return poses


def opengl_to_opencv_c2w(c2ws: np.ndarray) -> np.ndarray:
    """Convert Nerfstudio/OpenGL camera-to-world poses to OpenCV convention."""
    poses = _as_pose_array(c2ws)
    converted = poses @ _OPENGL_TO_OPENCV
    return converted.astype(np.float32)


def interpolate_camera_poses(
    c2ws: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate translation linearly and rotation with quaternion SLERP."""
    poses = _as_pose_array(c2ws)
    source = np.asarray(source_times, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    if source.ndim != 1 or source.shape[0] != poses.shape[0]:
        raise ValueError("source_times must be one-dimensional and match camera poses")
    if target.ndim != 1 or target.size == 0:
        raise ValueError("target_times must be a non-empty one-dimensional array")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("camera timestamps contain non-finite values")
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("source_times must contain at least two strictly increasing values")
    tolerance = 1e-9
    if target.min() < source[0] - tolerance or target.max() > source[-1] + tolerance:
        raise ValueError("target_times must stay within the source camera interval")

    translation = interp1d(
        source,
        poses[:, :3, 3],
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(target)
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat().copy()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], quaternions[index - 1]) < 0.0:
            quaternions[index] *= -1.0
    rotation = Slerp(source, Rotation.from_quat(quaternions))(target).as_matrix()

    output = np.zeros((len(target), 4, 4), dtype=np.float64)
    output[:, :3, :3] = rotation
    output[:, :3, 3] = translation
    output[:, 3, 3] = 1.0
    return output.astype(np.float32)


def _se3_inverse_fp32(transforms: torch.Tensor) -> torch.Tensor:
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3:]
    inverse_rotation = rotation.transpose(-1, -2)
    inverse_translation = -torch.bmm(inverse_rotation, translation)
    inverse = torch.eye(4, device=transforms.device, dtype=transforms.dtype)[
        None
    ].repeat(transforms.shape[0], 1, 1)
    inverse[:, :3, :3] = inverse_rotation
    inverse[:, :3, 3:] = inverse_translation
    return inverse


def relative_adjacent_poses(
    c2ws: np.ndarray,
    *,
    translation_eps: float = 1e-6,
) -> tuple[np.ndarray, float, bool]:
    """Return frame-adjacent c2w transforms with clip-max translation scaling.

    The first output pose is identity.  Remaining outputs transform a camera
    coordinate from the preceding latent frame into the current latent frame.
    The returned scale is the normalization divisor.  It is one for a static
    clip so numerical noise is never amplified.
    """
    if translation_eps <= 0:
        raise ValueError("translation_eps must be positive")
    poses = torch.from_numpy(_as_pose_array(c2ws).astype(np.float32, copy=True))
    first_w2c = _se3_inverse_fp32(poses[:1])
    adjacent = torch.matmul(first_w2c, poses)
    adjacent[0] = torch.eye(4, dtype=poses.dtype)
    if len(poses) > 1:
        adjacent[1:] = torch.bmm(
            _se3_inverse_fp32(adjacent[:-1]), adjacent[1:]
        )
    translations = adjacent[:, :3, 3]
    maximum_tensor = torch.norm(translations, dim=-1).max()
    maximum = float(maximum_tensor.item())
    static_camera = maximum <= translation_eps
    scale = 1.0 if static_camera else maximum
    if not static_camera:
        adjacent[:, :3, 3] = translations / maximum_tensor
    adjacent[0] = torch.eye(4, dtype=poses.dtype)
    return adjacent.numpy(), scale, static_camera


def transform_pixel_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_size: tuple[int, int],
    actual_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale pixel ``[fx, fy, cx, cy]`` through image scaling, crop and resize.

    Sizes use ``(width, height)`` order and the crop box uses PIL's
    ``(left, top, right, bottom)`` convention.
    """
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError("intrinsics must be finite [..., 4] arrays")
    source_width, source_height = (int(value) for value in source_size)
    actual_width, actual_height = (int(value) for value in actual_size)
    target_width, target_height = (int(value) for value in target_size)
    left, top, right, bottom = (int(value) for value in crop_box)
    if min(source_width, source_height, actual_width, actual_height) <= 0:
        raise ValueError("source and actual image sizes must be positive")
    if min(target_width, target_height, right - left, bottom - top) <= 0:
        raise ValueError("crop and target image sizes must be positive")

    output = values.copy()
    output[..., 0] *= actual_width / source_width
    output[..., 1] *= actual_height / source_height
    output[..., 2] *= actual_width / source_width
    output[..., 3] *= actual_height / source_height
    resize_x = target_width / float(right - left)
    resize_y = target_height / float(bottom - top)
    output[..., 0] *= resize_x
    output[..., 1] *= resize_y
    output[..., 2] = (output[..., 2] - left) * resize_x
    output[..., 3] = (output[..., 3] - top) * resize_y
    if np.any(output[..., :2] <= 0):
        raise ValueError("focal lengths must remain positive")
    return output.astype(np.float32)


def undistort_image_and_intrinsics(
    image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort an RGB image while preserving the same pixel intrinsics."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only in lean envs
        raise ImportError("OpenCV is required for DL3DV image undistortion") from exc

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {rgb.shape}")
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("intrinsics must be a finite [fx, fy, cx, cy] vector")
    coefficients = np.asarray(distortion, dtype=np.float64)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("distortion must be a finite [k1, k2, p1, p2, k3] vector")
    matrix = np.array(
        [[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if np.allclose(coefficients, 0.0, rtol=0.0, atol=0.0):
        return rgb.copy(), values.astype(np.float32)
    undistorted = cv2.undistort(rgb, matrix, coefficients, None, matrix)
    return undistorted, values.astype(np.float32)


def camera_origin_direction_rays(
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build upstream-compatible ``[..., origin(3), direction(3)]`` camera rays."""
    if c2ws.ndim != 3 or tuple(c2ws.shape[1:]) != (4, 4):
        raise ValueError(f"c2ws must have shape (F, 4, 4), got {tuple(c2ws.shape)}")
    if not c2ws.is_floating_point():
        raise TypeError("c2ws must use a floating-point dtype")
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0).expand(c2ws.shape[0], -1)
    if intrinsics.ndim != 2 or tuple(intrinsics.shape) != (c2ws.shape[0], 4):
        raise ValueError(
            "intrinsics must have shape (4,) or (F, 4), got "
            f"{tuple(intrinsics.shape)}"
        )
    if height <= 0 or width <= 0:
        raise ValueError("ray grid dimensions must be positive")
    intrinsics = intrinsics.to(device=c2ws.device, dtype=c2ws.dtype)
    if torch.any(intrinsics[:, :2] <= 0):
        raise ValueError("camera focal lengths must be positive")

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=c2ws.device, dtype=c2ws.dtype) + 0.5,
        torch.arange(width, device=c2ws.device, dtype=c2ws.dtype) + 0.5,
        indexing="ij",
    )
    x = grid_x.reshape(1, -1)
    y = grid_y.reshape(1, -1)
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    directions = torch.stack(
        [
            (x - cx[:, None]) / fx[:, None],
            (y - cy[:, None]) / fy[:, None],
            torch.ones((len(c2ws), height * width), device=c2ws.device, dtype=c2ws.dtype),
        ],
        dim=-1,
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    world_directions = torch.bmm(c2ws[:, :3, :3], directions.transpose(1, 2)).transpose(1, 2)
    origins = c2ws[:, :3, 3].unsqueeze(1).expand_as(world_directions)
    return torch.cat([origins, world_directions], dim=-1).reshape(
        len(c2ws), height, width, 6
    )


__all__ = [
    "camera_origin_direction_rays",
    "interpolate_camera_poses",
    "opengl_to_opencv_c2w",
    "relative_adjacent_poses",
    "transform_pixel_intrinsics",
    "undistort_image_and_intrinsics",
]
