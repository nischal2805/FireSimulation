"""Per-camera intrinsics from ChArUco views with seeded held-out validation (PRD 7.1).

Distortion model: OpenCV (k1, k2, p1, p2, k3); ``fix_k3`` fixes k3 = 0 (configurable). ``K_raw`` is the fitted
matrix for the distorted image; ``K`` (undistorted) = ``getOptimalNewCameraMatrix(K_raw, dist, size, alpha)``
with alpha 0 by default: the undistorted image has the same size and every pixel valid (edges cropped). Downstream
code uses the undistorted K (PRD 4). Pixels, host NumPy float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.calib.board import BoardSpec, detect_charuco, make_board, object_points, solve_board_pose

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class View:
    """One accepted board view: detected corners (N,2) px, ids (N,), source name."""

    corners: NDArray[np.float64]
    ids: NDArray[np.int32]
    name: str = ""


@dataclass
class IntrinsicsResult:
    """Outcome of an intrinsics fit. Errors are pixels."""

    cam_id: int
    width: int
    height: int
    K_raw: NDArray[np.float64]
    dist: NDArray[np.float64]
    K_undist: NDArray[np.float64]
    alpha: float
    rms_fit: float            # sqrt(mean sq) over fit views (re-evaluated with per-view PnP)
    rms_heldout: float        # same metric on the held-out views (never used for the fit)
    median_heldout: float     # median point error on the held-out views
    median_fit: float
    n_fit: int
    n_heldout: int
    fit_names: list[str] = field(default_factory=list)
    heldout_names: list[str] = field(default_factory=list)
    per_view_rms: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"cam{self.cam_id}: fx={self.K_raw[0, 0]:.1f} fy={self.K_raw[1, 1]:.1f} cx={self.K_raw[0, 2]:.1f} "
                f"cy={self.K_raw[1, 2]:.1f} dist={np.round(self.dist, 4).tolist()} | fit RMS {self.rms_fit:.3f} px "
                f"median {self.median_fit:.3f} ({self.n_fit} views) | HELD-OUT RMS {self.rms_heldout:.3f} px "
                f"median {self.median_heldout:.3f} ({self.n_heldout} views)")


def load_views_from_dir(spec: BoardSpec, folder: Path, min_corners: int) -> tuple[list[View], tuple[int, int], list[Path]]:
    """Detect the board in every image of ``folder`` (sorted). Returns (views, (w,h), images used).

    Images with fewer than ``min_corners`` corners are skipped; mixed resolutions raise."""
    views: list[View] = []
    used: list[Path] = []
    size: tuple[int, int] | None = None
    det = cv2.aruco.CharucoDetector(make_board(spec))
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        s = (img.shape[1], img.shape[0])
        if size is not None and s != size:
            raise ValueError(f"{p.name}: resolution {s} differs from the first image {size}")
        size = s
        c, ids = detect_charuco(spec, img, detector=det)
        if len(ids) >= min_corners:
            views.append(View(c, ids, p.name))
            used.append(p)
    if size is None:
        raise ValueError(f"no readable images in {folder}")
    return views, size, used


def split_views(n: int, holdout_fraction: float, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Seeded split of ``n`` view indices into (fit, held_out); at least 1 held out and 8 fit."""
    n_hold = max(1, round(holdout_fraction * n))
    if n - n_hold < 8:
        raise ValueError(f"too few views ({n}) for a held-out split of {n_hold}")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])


def _view_errors(spec: BoardSpec, v: View, K: NDArray, dist: NDArray) -> NDArray[np.float64]:
    """Per-corner reprojection error (N,) px of a view under the camera, with a per-view PnP pose."""
    T, _ = solve_board_pose(spec, v.corners, v.ids, K, dist)
    obj = object_points(spec, v.ids)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    proj, _ = cv2.projectPoints(obj, rvec, T[:3, 3], K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - v.corners, axis=1)


def fit_intrinsics(
    spec: BoardSpec,
    views: list[View],
    size: tuple[int, int],
    *,
    cam_id: int = 0,
    min_views: int = 20,
    holdout_fraction: float = 0.2,
    seed: int = 1234,
    fix_k3: bool = True,
    alpha: float = 0.0,
) -> IntrinsicsResult:
    """Fit intrinsics on a seeded subset of ``views`` and validate on the rest.

    ``size`` = (width, height). Raises ValueError if fewer than ``min_views`` views."""
    if len(views) < min_views:
        raise ValueError(f"only {len(views)} usable views, need >= {min_views}")
    fit_i, hold_i = split_views(len(views), holdout_fraction, seed)
    obj = [object_points(spec, views[i].ids).astype(np.float32) for i in fit_i]
    img = [views[i].corners.astype(np.float32).reshape(-1, 1, 2) for i in fit_i]
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    _, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    dist = np.asarray(dist, np.float64).reshape(-1)[:5]
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    K = np.asarray(K, np.float64)
    fit_err = [_view_errors(spec, views[i], K, dist) for i in fit_i]
    hold_err = [_view_errors(spec, views[i], K, dist) for i in hold_i]
    fe, he = np.concatenate(fit_err), np.concatenate(hold_err)
    K_u, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    per_view = {views[i].name or str(i): float(np.sqrt(np.mean(e**2))) for i, e in zip([*fit_i, *hold_i], [*fit_err, *hold_err])}
    return IntrinsicsResult(
        cam_id, size[0], size[1], K, dist, np.asarray(K_u, np.float64), alpha,
        float(np.sqrt(np.mean(fe**2))), float(np.sqrt(np.mean(he**2))), float(np.median(he)), float(np.median(fe)),
        len(fit_i), len(hold_i), [views[i].name for i in fit_i], [views[i].name for i in hold_i], per_view,
    )


# ---------------------------------------------------------------------------- interactive capture logic (pure)


def pose_distance(a: View, b: View, diag: float) -> float:
    """Image-space difference between two board views: mean displacement of the shared corners / image diagonal.

    Fewer than 4 shared corner ids counts as a different pose (1.0)."""
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 4:
        return 1.0
    return float(np.mean(np.linalg.norm(a.corners[ia] - b.corners[ib], axis=1)) / diag)


class CoverageMap:
    """Counts of detected corners per image cell: ``grid=(nx, ny)`` over ``size=(w, h)``."""

    def __init__(self, size: tuple[int, int], grid: tuple[int, int] = (8, 6)) -> None:
        self.size, self.grid = size, grid
        self.counts = np.zeros((grid[1], grid[0]), np.int32)

    def add(self, corners: NDArray) -> None:
        if len(corners) == 0:
            return
        gx = np.clip((corners[:, 0] / self.size[0] * self.grid[0]).astype(int), 0, self.grid[0] - 1)
        gy = np.clip((corners[:, 1] / self.size[1] * self.grid[1]).astype(int), 0, self.grid[1] - 1)
        np.add.at(self.counts, (gy, gx), 1)

    @property
    def covered_fraction(self) -> float:
        """Fraction of cells with at least one corner."""
        return float(np.mean(self.counts > 0))

    def heatmap(self) -> NDArray[np.uint8]:
        """(h, w, 3) BGR heat-map image at the full image size (cold = no corners)."""
        m = self.counts.astype(np.float32)
        m = m / max(float(m.max()), 1.0)
        small = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
        small[self.counts == 0] = (60, 20, 20)
        return cv2.resize(small, self.size, interpolation=cv2.INTER_NEAREST)


class CaptureGate:
    """Auto-capture decision: board still for ``still_frames`` frames and pose different from all captures."""

    def __init__(self, size: tuple[int, int], min_corners: int, still_px: float, still_frames: int,
                 min_pose_diff: float) -> None:
        self.diag = float(np.hypot(*size))
        self.min_corners, self.still_px, self.still_frames = min_corners, still_px, still_frames
        self.min_pose_diff = min_pose_diff
        self.captured: list[View] = []
        self._prev: View | None = None
        self._still = 0

    def update(self, view: View | None) -> View | None:
        """Feed the newest detection (or None); returns the view to capture now, else None."""
        if view is None or len(view.ids) < self.min_corners:
            self._prev, self._still = None, 0
            return None
        if self._prev is not None and pose_distance(self._prev, view, 1.0) < self.still_px:  # diag=1: raw px
            self._still += 1
        else:
            self._still = 0
        self._prev = view
        if self._still < self.still_frames:
            return None
        if any(pose_distance(c, view, self.diag) < self.min_pose_diff for c in self.captured):
            return None
        self.captured.append(view)
        self._still = 0
        return view
