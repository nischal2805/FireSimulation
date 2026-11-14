"""Geometry contract (PRD section 4): cameras, transforms, depth, AABB, voxel grid.

Conventions: world right-handed, metres, +Z up; camera frame OpenCV (+X right, +Y down,
+Z forward); transforms are named ``T_<to>_from_<from>`` (4x4, column vectors);
integer pixel coordinates (u, v) denote pixel centres. All maths is float64 NumPy on the
host; functions are vectorised over N and return new arrays (callers own the results).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def make_K(fx: float, fy: float, cx: float, cy: float) -> Array:
    """Pinhole intrinsics, shape (3,3), pixels."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def invert_transform(T: ArrayLike) -> Array:
    """Rigid inverse of a 4x4 transform (R^T, -R^T t); not ``np.linalg.inv``."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def transform_points(T_b_from_a: ArrayLike, p_a: ArrayLike) -> Array:
    """Apply a 4x4 transform to points (N,3) in frame a, returning (N,3) in frame b."""
    T = np.asarray(T_b_from_a, dtype=np.float64)
    p = np.asarray(p_a, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def look_at(position: ArrayLike, target: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)) -> Array:
    """T_world_from_camera for a camera at ``position`` looking at ``target`` (OpenCV axes).

    If the view direction is parallel to ``up`` (top-down), world +Y is used as up instead."""
    pos = np.asarray(position, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        # Looking straight along +-up (e.g. top-down camera): fall back to world +Y as up.
        x = np.cross(z, (0.0, 1.0, 0.0))
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


@dataclass(frozen=True)
class CameraCalibration:
    """Calibrated camera. ``K`` is the undistorted (3,3) intrinsics in pixels for
    ``width`` x ``height``; ``dist`` is (5,) OpenCV distortion; ``T_world_from_camera``
    is (4,4) camera->world."""

    cam_id: int
    width: int
    height: int
    K: Array
    dist: Array
    T_world_from_camera: Array
    calib_version: int

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        dist = np.asarray(self.dist, dtype=np.float64)
        T = np.asarray(self.T_world_from_camera, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be (3,3), got {K.shape}")
        if dist.shape != (5,):
            raise ValueError(f"dist must be (5,), got {dist.shape}")
        if T.shape != (4, 4):
            raise ValueError(f"T_world_from_camera must be (4,4), got {T.shape}")
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("T_world_from_camera bottom row must be [0,0,0,1]")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or abs(np.linalg.det(R) - 1.0) > 1e-6:
            raise ValueError("rotation must be orthonormal with det +1")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist", dist)
        object.__setattr__(self, "T_world_from_camera", T)

    @property
    def T_camera_from_world(self) -> Array:
        """(4,4) world->camera transform."""
        return invert_transform(self.T_world_from_camera)

    @property
    def center_world(self) -> Array:
        """Camera centre in world coordinates, (3,) metres."""
        return self.T_world_from_camera[:3, 3].copy()


def project(calib: CameraCalibration, p_world: ArrayLike) -> tuple[Array, Array, NDArray[np.bool_]]:
    """Project world points (N,3) m -> (uv (N,2) px, z (N,) m camera Z, valid (N,) bool).

    Points with z <= 0 get uv = 0.0 and valid = False (never NaN). Distortion ignored (K is
    the undistorted intrinsics)."""
    pc = transform_points(calib.T_camera_from_world, np.atleast_2d(p_world))
    z = pc[:, 2]
    valid = z > 0.0
    zs = np.where(valid, z, 1.0)
    uvw = pc @ calib.K.T
    uv = uvw[:, :2] / zs[:, None]
    uv = np.where(valid[:, None], uv, 0.0)
    return uv, z, valid


def backproject(calib: CameraCalibration, uv: ArrayLike, z: ArrayLike) -> Array:
    """World points (N,3) from pixels uv (N,2) and Z-depth z (N,) m:
    ``p_c = z * inv(K) @ [u,v,1]``, ``p_w = T_world_from_camera @ [p_c,1]``."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    pc = z[:, None] * (uv1 @ np.linalg.inv(calib.K).T)
    return transform_points(calib.T_world_from_camera, pc)


def pixel_grid(width: int, height: int) -> tuple[Array, Array]:
    """Integer pixel-centre grids (u, v), each (height, width), float64 values."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    return u, v


def backproject_depth(calib: CameraCalibration, depth: ArrayLike, valid: ArrayLike) -> Array:
    """Z-depth map (H,W) m + valid (H,W) bool -> world points (H,W,3); invalid pixels = 0.0."""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    h, w = depth.shape
    u, v = pixel_grid(w, h)
    uv = np.stack([u.ravel(), v.ravel()], axis=1)
    pts = backproject(calib, uv, np.where(valid, depth, 0.0).ravel()).reshape(h, w, 3)
    return np.where(valid[..., None], pts, 0.0)


def ray_dirs_camera(calib: CameraCalibration) -> Array:
    """Per-pixel camera-frame ray directions (H,W,3) = inv(K) @ [u,v,1] (z component 1)."""
    u, v = pixel_grid(calib.width, calib.height)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
    return uv1 @ np.linalg.inv(calib.K).T


def _ray_norm(calib: CameraCalibration) -> Array:
    return np.linalg.norm(ray_dirs_camera(calib), axis=-1)


def zdepth_to_raydist(calib: CameraCalibration, z: ArrayLike) -> Array:
    """Z-depth (H,W) m -> ray distance (H,W) m, ``r = z * |inv(K)[u,v,1]|``; 0.0 stays 0.0."""
    return np.asarray(z, dtype=np.float64) * _ray_norm(calib)


def raydist_to_zdepth(calib: CameraCalibration, r: ArrayLike) -> Array:
    """Ray distance (H,W) m -> Z-depth (H,W) m, ``z = r / |inv(K)[u,v,1]|``; 0.0 stays 0.0."""
    return np.asarray(r, dtype=np.float64) / _ray_norm(calib)


@dataclass(frozen=True)
class AABB:
    """Axis-aligned box, world frame, metres; ``min``/``max`` are (3,)."""

    min: Array
    max: Array

    def __post_init__(self) -> None:
        mn = np.asarray(self.min, dtype=np.float64)
        mx = np.asarray(self.max, dtype=np.float64)
        if mn.shape != (3,) or mx.shape != (3,):
            raise ValueError("AABB min/max must be (3,)")
        if np.any(mx <= mn):
            raise ValueError("AABB requires max > min on every axis")
        object.__setattr__(self, "min", mn)
        object.__setattr__(self, "max", mx)

    @property
    def size(self) -> Array:
        return self.max - self.min

    @property
    def center(self) -> Array:
        return 0.5 * (self.min + self.max)

    def contains(self, p: ArrayLike) -> NDArray[np.bool_]:
        """(N,3) points -> (N,) bool, inclusive of the faces."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        return np.all((p >= self.min) & (p <= self.max), axis=1)

    def ray_intersect(
        self, origins: ArrayLike, dirs: ArrayLike
    ) -> tuple[Array, Array, NDArray[np.bool_]]:
        """Slab method. origins, dirs (N,3) -> (t_near (N,), t_far (N,), hit (N,)).

        Ray is ``o + t d`` with t >= 0; for an origin inside the box t_near is clamped to 0.
        Misses (and boxes entirely behind the origin) have hit=False and t_near = t_far = 0."""
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        d = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
        parallel = d == 0.0
        safe = np.where(parallel, 1.0, d)
        t1 = (self.min - o) / safe
        t2 = (self.max - o) / safe
        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        inside = (o >= self.min) & (o <= self.max)
        # Parallel axis: no constraint if origin within the slab, otherwise a miss.
        lo = np.where(parallel, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(parallel, np.where(inside, np.inf, -np.inf), hi)
        t_near = np.maximum(lo.max(axis=1), 0.0)
        t_far = hi.min(axis=1)
        hit = t_far >= t_near
        t_near = np.where(hit, t_near, 0.0)
        t_far = np.where(hit, t_far, 0.0)
        return t_near, t_far, hit


@dataclass(frozen=True)
class VoxelGrid:
    """Cubic ROI split into N^3 voxels; centre of voxel i is ``roi.min + (i + 0.5) * h``."""

    roi: AABB
    N: int

    def __post_init__(self) -> None:
        s = self.roi.size
        if not np.allclose(s, s[0], rtol=1e-9, atol=1e-12):
            raise ValueError(f"VoxelGrid requires a cubic ROI, got size {s}")
        if self.N < 1:
            raise ValueError("N must be >= 1")

    @property
    def h(self) -> float:
        """Voxel edge length, metres."""
        return float(self.roi.size[0] / self.N)

    def centers(self, idx: ArrayLike) -> Array:
        """Integer indices (M,3) -> voxel centres (M,3) m."""
        idx = np.atleast_2d(np.asarray(idx))
        return self.roi.min + (idx + 0.5) * self.h

    def index_of(self, p: ArrayLike) -> NDArray[np.int64]:
        """Points (M,3) m -> integer voxel indices (M,3) (floor; may be out of bounds)."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        return np.floor((p - self.roi.min) / self.h).astype(np.int64)

    def in_bounds(self, idx: ArrayLike) -> NDArray[np.bool_]:
        """(M,3) indices -> (M,) bool, True where 0 <= i < N on all axes."""
        idx = np.atleast_2d(np.asarray(idx))
        return np.all((idx >= 0) & (idx < self.N), axis=1)

    def linear_index(self, idx: ArrayLike) -> NDArray[np.int64]:
        """(M,3) indices (ix,iy,iz) -> (M,) C-order linear indices."""
        idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
        return (idx[:, 0] * self.N + idx[:, 1]) * self.N + idx[:, 2]

    def unravel(self, lin: ArrayLike) -> NDArray[np.int64]:
        """(M,) C-order linear indices -> (M,3) indices (ix,iy,iz)."""
        lin = np.atleast_1d(np.asarray(lin, dtype=np.int64))
        return np.stack(np.unravel_index(lin, (self.N,) * 3), axis=1).astype(np.int64)
