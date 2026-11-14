"""Synthetic ChArUco renderer with exact ground truth (tests; no camera needed).

Inverse-warps the printed board into a camera with known ``K``, OpenCV ``dist`` and pose: each output pixel is
undistorted to a normalised ray (``cv2.undistortPoints``, the inverse of ``cv2.projectPoints``), intersected with the
board plane z = 0 and sampled from the board image, so the rendering is consistent with ``projectPoints`` to
sub-pixel accuracy. Seeded blur/noise. World frame per P1R2; metres; uint8 gray images.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.calib.board import BoardSpec, T_world_from_board, make_board
from livefire.common.geometry import invert_transform, look_at


class SyntheticBoardCamera:
    """Renders board views for one camera model. Rays and the board image are cached across poses.

    ``K`` (3,3) px and ``dist`` (5,) describe the *distorted* image of ``size`` = (w, h)."""

    def __init__(self, spec: BoardSpec, K: NDArray, dist: NDArray, size: tuple[int, int], *,
                 px_per_m: float = 1600.0, margin_m: float = 0.03) -> None:
        self.spec, self.K, self.dist, self.size = spec, np.asarray(K, np.float64), np.asarray(dist, np.float64), size
        self.px_per_m, self.margin_m = px_per_m, margin_m
        bw, bh = spec.size_m
        w, h = round(bw * px_per_m), round(bh * px_per_m)
        img = make_board(spec).generateImage((w, h), marginSize=0, borderBits=1)
        m = round(margin_m * px_per_m)
        self.board_img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
        self._m = m
        u, v = np.meshgrid(np.arange(size[0], dtype=np.float64), np.arange(size[1], dtype=np.float64))
        pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        n = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        self._rays = np.concatenate([n, np.ones((len(n), 1))], axis=1)  # camera-frame ray, z = 1
        self._T_board_from_world = invert_transform(T_world_from_board(spec))

    def render(self, T_world_from_camera: NDArray, rng: np.random.Generator, *, noise_sigma: float = 1.5,
               blur_sigma: float = 0.7) -> NDArray[np.uint8]:
        """(h, w) uint8 image of the board seen from ``T_world_from_camera`` (4,4, camera->world)."""
        T = self._T_board_from_world @ np.asarray(T_world_from_camera, np.float64)
        d = self._rays @ T[:3, :3].T
        o = T[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -o[2] / d[:, 2]
        ok = s > 0
        X = o[0] + s * d[:, 0]
        Y = o[1] + s * d[:, 1]
        mapx = np.where(ok, (X + self.margin_m) * self.px_per_m - 0.5, -1e4).astype(np.float32)
        mapy = np.where(ok, (Y + self.margin_m) * self.px_per_m - 0.5, -1e4).astype(np.float32)
        w, h = self.size
        out = cv2.remap(self.board_img, mapx.reshape(h, w), mapy.reshape(h, w), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=170)
        out = out.astype(np.float32)
        if blur_sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
        if noise_sigma > 0:
            out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)


def random_view_poses(n: int, rng: np.random.Generator, *, distance: tuple[float, float] = (0.55, 0.9),
                      elevation_deg: tuple[float, float] = (35.0, 80.0), max_roll_deg: float = 25.0,
                      target_jitter_m: float = 0.12) -> list[NDArray[np.float64]]:
    """``n`` camera poses T_world_from_camera (4,4) looking at the board region, varied tilt, roll and offset."""
    poses = []
    for _ in range(n):
        az = rng.uniform(0, 2 * np.pi)
        el = np.deg2rad(rng.uniform(*elevation_deg))
        r = rng.uniform(*distance)
        pos = r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        tgt = np.array([rng.uniform(-target_jitter_m, target_jitter_m), rng.uniform(-target_jitter_m, target_jitter_m), 0.0])
        T = look_at(pos, tgt)
        a = np.deg2rad(rng.uniform(-max_roll_deg, max_roll_deg))
        c, s = np.cos(a), np.sin(a)
        Rz = np.eye(4)
        Rz[:2, :2] = [[c, -s], [s, c]]
        poses.append(T @ Rz)  # roll about the camera optical axis
    return poses
