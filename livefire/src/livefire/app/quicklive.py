"""Quick live mode: ONE uncalibrated camera (phone IP cam / webcam), no ChArUco board.

Instead of a calibration the world is derived from the first frame: intrinsics come from an assumed horizontal field
of view, metric monocular depth is back-projected, the dominant plane (the table, or the floor in a top-down view) is
fitted by RANSAC and defines the world (origin = plane inlier centroid, +Z = plane normal toward the camera, +X = the
camera's right axis projected on the plane). Depth is then rescaled per frame so the plane sits at z = 0 and the normal
acquisition path (accumulate -> TSDF -> objects -> snapshot) runs unchanged. Accuracy is approximate by design
(assumed FOV, monocular depth); keep the phone still on a stand. Frames: world +Z up (plane normal), camera OpenCV.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from livefire.app.hud import make_ladder
from livefire.app.lifecycle import Lifecycle, State
from livefire.app.live import LiveSession
from livefire.app.window import fit_limits, run_session
from livefire.capture.factory import make_source
from livefire.common.config import load_yaml
from livefire.common.errors import DepthError, ReconstructionError
from livefire.common.geometry import CameraCalibration, invert_transform, make_K
from livefire.common.log import get_logger
from livefire.perception.depth.base import depth_result_from_z
from livefire.scene.acquire import acquire_with_report

log = get_logger("livefire.app.quicklive")


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole K (3,3) px for a ``width`` x ``height`` image with horizontal field of view ``hfov_deg``; square pixels,
    principal point at the image centre."""
    f = 0.5 * width / np.tan(0.5 * np.radians(hfov_deg))
    return make_K(f, f, 0.5 * (width - 1), 0.5 * (height - 1))


def pixel_rays(K: np.ndarray, h: int, w: int) -> np.ndarray:
    """(h, w, 3) camera-frame ray directions with z = 1 for every pixel centre."""
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)


def fit_plane_ransac(
    z: np.ndarray, rays: np.ndarray, valid: np.ndarray, *, iters: int = 400, thresh_rel: float = 0.02,
    up_pref: np.ndarray | None = None, seed: int = 1234,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Dominant plane of the back-projected depth. ``z`` (h,w) m, ``rays`` (h,w,3), ``valid`` (h,w) bool.

    Returns ``(n, d, inlier_mask)``: unit normal ``n`` in camera coordinates pointing TOWARD the camera, offset ``d``
    (< 0) with ``n . x = d`` on the plane, and the (h,w) inlier mask (distance < ``thresh_rel`` * depth). Candidates
    whose normal is far from ``up_pref`` (camera-frame 'up', e.g. (0,-1,0)) are down-weighted x0.3 so a facing wall
    does not beat a table; with ``up_pref=None`` the plane with the most inliers wins. Raises ``ReconstructionError``."""
    rng = np.random.default_rng(seed)
    pts_all = rays * z[..., None]
    idx = np.flatnonzero(valid.ravel())
    if len(idx) < 500:
        raise ReconstructionError("too few valid depth pixels to fit a plane")
    sub = rng.choice(idx, size=min(len(idx), 20000), replace=False)
    P = pts_all.reshape(-1, 3)[sub]
    tol = thresh_rel * P[:, 2]
    best, best_score = None, -1.0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = float(n @ a)
        if d > 0:
            n, d = -n, -d
        cnt = float((np.abs(P @ n - d) < tol).sum())
        wgt = 1.0 if up_pref is None or float(n @ up_pref) > 0.25 else 0.3
        if cnt * wgt > best_score:
            best, best_score = (n, d), cnt * wgt
    if best is None:
        raise ReconstructionError("plane RANSAC failed")
    n, d = best
    for _ in range(3):  # least-squares refinement on the inliers
        m = np.abs(P @ n - d) < tol
        if m.sum() < 50:
            break
        Q = P[m]
        c0 = Q.mean(axis=0)
        n = np.linalg.svd(Q - c0, full_matrices=False)[2][-1]
        n = n / np.linalg.norm(n)
        d = float(n @ c0)
        if d > 0:
            n, d = -n, -d
    dist = np.abs(pts_all @ n - d)
    mask = valid & (dist < thresh_rel * np.maximum(z, 1e-3))
    return n, d, mask


def world_from_plane(n: np.ndarray, centroid_cam: np.ndarray) -> np.ndarray:
    """``T_world_from_camera`` (4,4) for the plane world: +Z = ``n`` (toward the camera), origin = ``centroid_cam``
    (a plane point, camera coords), +X = camera right axis projected onto the plane, right-handed."""
    e1 = np.array([1.0, 0.0, 0.0])
    x = e1 - (e1 @ n) * n
    if np.linalg.norm(x) < 1e-6:  # camera x parallel to the normal: degenerate, use camera up
        up = np.array([0.0, -1.0, 0.0])
        x = up - (up @ n) * n
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    T_cam_from_world = np.eye(4)
    T_cam_from_world[:3, :3] = np.stack([x, y, n], axis=1)
    T_cam_from_world[:3, 3] = centroid_cam
    return invert_transform(T_cam_from_world)


class PlaneAnchoredDepth:
    """Metric depth backend whose output is rescaled per frame so the table plane sits exactly at world z = 0.

    ``raw`` is a metric ``HFDepthBackend``; ``plane_z`` (h,w) is the analytic plane Z-depth (m) and ``plane_mask``
    (h,w) bool the plane pixels, both fixed at the first-frame fit. Scale = median(plane_z / raw) over the mask."""

    def __init__(self, raw: Any, plane_z: np.ndarray, plane_mask: np.ndarray) -> None:
        self.raw, self.plane_z, self.plane_mask = raw, plane_z, plane_mask
        self.scale = 1.0

    def infer(self, frame: Any) -> Any:
        """Anchored ``DepthResult`` for ``frame`` (same size as ``plane_z``)."""
        r = self.raw.infer_raw(frame)
        m = self.plane_mask & r.valid
        if m.sum() < 200:
            raise DepthError("table plane not visible enough in this frame to anchor depth")
        self.scale = float(np.median(self.plane_z[m] / r.values[m]))
        return depth_result_from_z(r.values * self.scale, r.valid, 0.5, frame.calib_version)


class _Identity:
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return rgb


class QuickLiveSession(LiveSession):
    """``LiveSession`` for one uncalibrated camera: world derived from the dominant plane (see module doc)."""

    def __init__(self, source: str | int, hfov_deg: float = 65.0, depth: str | None = None, grid: int | None = None,
                 max_w: int = 1600, max_h: int = 900, top_down: bool = False, width: int = 1280,
                 height: int = 720) -> None:
        self.scene, self.grid_override = "live", ({"G": grid} if grid else {})
        self.app = load_yaml("app")
        self.fps = float(self.app.get("fps", 30))
        self.max_w, self.max_h = max_w, max_h
        self.depth_name, self.hfov, self.top_down = depth, hfov_deg, top_down
        self.life = Lifecycle()
        self._last: dict[int, Any] = {}
        self.sources: list[Any] = []
        self._raw: Any = None
        self.cal = SimpleNamespace(version=1)
        self.undist = {0: _Identity()}
        is_url = isinstance(source, str) and not source.isdigit()
        cfg: dict[str, Any] = {"cam_id": 0, "type": "url" if is_url else "usb",
                               "source": source if is_url else int(source),
                               "width": width, "height": height, "fps": 30, "backend": "auto"}
        if not is_url:
            cfg["fourcc"] = "MJPG"
        try:
            self.life.transition(State.CALIBRATING)
            src = make_source(cfg, 0)
            self.sources = [src]
            src.start()
            self.life.transition(State.ACQUIRING)
            self._acquire_and_build()
            self.life.transition(State.READY)
        except Exception:
            self.life.transition(State.ERROR)
            self.close()
            raise
        self.ladder = make_ladder(float(self.app["degrade_volume_scale"]), float(self.app["degrade_step_factor"]))
        self._stamps: deque[float] = deque(maxlen=31)
        self.rt = 0.0

    def _first_frame(self, timeout_s: float = 20.0) -> Any:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            f = self.sources[0].latest()
            if f is not None:
                return f
            time.sleep(0.05)
        raise ReconstructionError("no frame from the camera (check the URL and that the phone app is streaming)")

    def _acquire_and_build(self) -> None:
        from livefire.perception.depth.hf_backends import create_depth_backend

        if self._raw is None:
            self._raw = create_depth_backend(self.depth_name)
            if self._raw.kind != "metric":
                raise DepthError("quick live needs a metric depth backend (da2_metric_small)")
        f0 = self._first_frame()
        h, w = f0.rgb.shape[:2]
        K = intrinsics_from_fov(w, h, self.hfov)
        rays = pixel_rays(K, h, w)
        log.info("estimating the table plane from the first frame (assumed horizontal FOV %.0f deg)...", self.hfov)
        raw0 = self._raw.infer_raw(f0)
        z0 = raw0.values.astype(np.float64)
        up = None if self.top_down else np.array([0.0, -1.0, 0.0])
        n, d, mask = fit_plane_ransac(z0, rays, raw0.valid, up_pref=up)
        if mask.mean() < 0.05:
            raise ReconstructionError(f"dominant plane covers only {100 * mask.mean():.1f}% of the image: point the "
                                      "camera so the table fills a good part of the view")
        cen = (rays[mask] * z0[mask][:, None]).mean(axis=0)
        cen = cen - (n @ cen - d) * n  # onto the plane
        T_wc = world_from_plane(n, cen)
        cal = CameraCalibration(0, w, h, K, np.zeros(5), T_wc, 1)
        log.info("table plane: %.0f%% of pixels, camera height %.2f m, optical axis %.0f deg from the plane normal",
                 100 * mask.mean(), T_wc[2, 3], np.degrees(np.arccos(np.clip(-n[2], -1, 1))))
        denom = rays @ n
        plane_z = (d / np.where(np.abs(denom) < 1e-6, 1e-6, denom)).astype(np.float32)  # plane Z-depth per pixel
        backend = PlaneAnchoredDepth(self._raw, plane_z, mask)
        log.info("acquiring the scene (keep the phone and the scene still)...")
        res = acquire_with_report(self.sources, {0: cal}, {0: backend},
                                  grid_cfg={**load_yaml("grid"), **self.grid_override})
        self._finish_build(res)


def run_quicklive(source: str, hfov: float, depth: str | None, grid: int | None, out: Path, top_down: bool,
                  max_w: int | None = None, max_h: int | None = None, frames: int | None = None) -> int:
    """``python -m livefire.app quicklive --source URL``: derive the world from the camera view, then run the window."""
    max_w, max_h = fit_limits(max_w, max_h)
    t0 = time.perf_counter()
    sess = QuickLiveSession(source, hfov, depth, grid, max_w, max_h, top_down)
    log.info("ready after %.1f s: left click ignites, right click selects, 1-8 sets material, o overlay, "
             "r re-acquire, q quit", time.perf_counter() - t0)
    try:
        return run_session(sess, "quicklive", out, frames, False)
    finally:
        sess.close()
