"""Extrinsics: per-camera board pose -> ``T_world_from_camera`` in the P1R2 world frame, with quality checks.

The board lies flat on the table, visible from every camera. Per camera ``solvePnP`` on the ChArUco corners (with the
fitted K/dist) gives ``T_camera_from_board``; ``T_world_from_camera = T_world_from_board @ T_board_from_camera``
(origin board centre, +X board X, +Z toward the cameras, table plane z = 0). Metres, host NumPy float64.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.calib.board import (
    BoardSpec,
    CalibrationError,
    T_world_from_board,
    board_corners_world,
    solve_board_pose,
)
from livefire.calib.intrinsics import IntrinsicsResult, View
from livefire.common.geometry import invert_transform


@dataclass
class CameraExtrinsics:
    """Averaged world pose of one camera. ``spread_mm`` / ``spread_deg`` = RMS deviation of the kept per-frame poses
    from the averaged pose (translation, rotation angle); 0 for a single frame."""

    cam_id: int
    T_world_from_camera: NDArray[np.float64]
    reproj_rms_px: float
    n_corners: int
    n_frames: int = 1
    spread_mm: float = 0.0
    spread_deg: float = 0.0


@dataclass
class ExtrinsicsResult:
    cameras: list[CameraExtrinsics]
    consistency_m: dict[tuple[int, int], float] = field(default_factory=dict)  # median triangulated-vs-layout dist
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"cam{c.cam_id}: reproj RMS {c.reproj_rms_px:.3f} px ({c.n_corners} corners, {c.n_frames} frame(s)), "
                 f"position {np.round(c.T_world_from_camera[:3, 3], 4).tolist()} m, "
                 f"pose spread {c.spread_mm:.2f} mm / {c.spread_deg:.3f} deg" for c in self.cameras]
        lines += [f"cams {a}-{b}: triangulated board corners vs layout, median {d * 1000:.2f} mm"
                  for (a, b), d in self.consistency_m.items()]
        return "\n".join(lines + [f"WARNING: {w}" for w in self.warnings])


def _mean_rotation(Rs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Chordal L2 mean of rotation matrices (n,3,3): SVD of the summed matrices projected back onto SO(3)."""
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def _rot_angle_deg(Ra: NDArray, Rb: NDArray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0))))


def _average_poses(Ts: list[NDArray[np.float64]]) -> tuple[NDArray[np.float64], float, float]:
    """Mean of 4x4 poses: SVD rotation mean + mean translation; returns (T, spread_mm, spread_deg) (RMS deviations)."""
    R = _mean_rotation(np.stack([T[:3, :3] for T in Ts]))
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    dt = np.array([np.linalg.norm(Ti[:3, 3] - t) for Ti in Ts])
    da = np.array([_rot_angle_deg(R, Ti[:3, :3]) for Ti in Ts])
    return T, float(np.sqrt(np.mean(dt**2)) * 1000.0), float(np.sqrt(np.mean(da**2)))


def estimate_extrinsics(
    spec: BoardSpec,
    intrinsics: dict[int, IntrinsicsResult],
    views: dict[int, View | Sequence[View]],
    *,
    min_corners: int = 8,
    max_reproj_rms_px: float = 1.5,
    max_consistency_m: float = 0.005,
) -> ExtrinsicsResult:
    """World poses of every camera in ``views`` (cam_id -> one detection or a list of detections of the static
    board, in the distorted image of that camera).

    Per camera: PnP per frame, frames with reprojection RMS > 2x the per-camera median are rejected, rotations are
    averaged (SVD of the summed matrices), translations by their mean; the pose spread is reported.
    Raises ValueError when a camera has no frame with >= ``min_corners`` corners, and CalibrationError when a camera
    ends up below the table plane (board seen from the wrong side). Warnings (not errors) for reprojection or
    consistency above the limits."""
    T_wb = T_world_from_board(spec)
    cams: list[CameraExtrinsics] = []
    kept_view: dict[int, View] = {}
    warns: list[str] = []
    for cid, vs in sorted(views.items()):
        frames = [vs] if isinstance(vs, View) else list(vs)
        usable = [v for v in frames if len(v.ids) >= min_corners]
        if not usable:
            seen = max((len(v.ids) for v in frames), default=0)
            raise ValueError(f"cam{cid}: only {seen} board corners visible (need >= {min_corners}); "
                             "move the board or the camera so the whole board is in view")
        it = intrinsics[cid]
        sols = [solve_board_pose(spec, v.corners, v.ids, it.K_raw, it.dist) for v in usable]
        rms = np.array([r for _, r in sols])
        keep = [i for i in range(len(usable)) if len(usable) == 1 or rms[i] <= 2.0 * float(np.median(rms))]
        Ts = [T_wb @ invert_transform(sols[i][0]) for i in keep]
        T, spread_mm, spread_deg = _average_poses(Ts)
        best = keep[int(np.argmin(rms[keep]))]
        kept_view[cid] = usable[best]
        r = float(np.sqrt(np.mean(rms[keep] ** 2)))
        cams.append(CameraExtrinsics(cid, T, r, len(usable[best].ids), len(keep), spread_mm, spread_deg))
        if len(keep) < len(usable):
            warns.append(f"cam{cid}: rejected {len(usable) - len(keep)} of {len(usable)} frames (reprojection > 2x median)")
        if r > max_reproj_rms_px:
            warns.append(f"cam{cid}: board reprojection RMS {r:.2f} px > {max_reproj_rms_px} px")
    below = [(c.cam_id, round(float(c.T_world_from_camera[2, 3]), 3)) for c in cams if c.T_world_from_camera[2, 3] < 0]
    if below:
        raise CalibrationError(
            f"camera(s) {below} (cam id, world z in m) are below the table plane: the board was seen from the wrong side "
            "(or from below). Lay the board flat, printed face up, with every camera looking down at it from above, "
            "and capture again.")

    cons: dict[tuple[int, int], float] = {}
    layout = board_corners_world(spec)
    for a, b in combinations(cams, 2):
        va, vb = kept_view[a.cam_id], kept_view[b.cam_id]
        common, ia, ib = np.intersect1d(va.ids, vb.ids, return_indices=True)
        if len(common) < 4:
            warns.append(f"cams {a.cam_id}-{b.cam_id}: only {len(common)} shared corners, consistency not checked")
            continue
        P = []
        pts = []
        for c, v, idx in ((a, va, ia), (b, vb, ib)):
            it = intrinsics[c.cam_id]
            Pc = it.K_undist @ invert_transform(c.T_world_from_camera)[:3]
            P.append(Pc)
            u = cv2.undistortPoints(v.corners[idx].reshape(-1, 1, 2), it.K_raw, it.dist, P=it.K_undist)
            pts.append(u.reshape(-1, 2).T)
        X = cv2.triangulatePoints(P[0], P[1], pts[0], pts[1])
        X = (X[:3] / X[3]).T
        d = np.linalg.norm(X - layout[common], axis=1)
        cons[(a.cam_id, b.cam_id)] = float(np.median(d))
        if cons[(a.cam_id, b.cam_id)] > max_consistency_m:
            warns.append(f"cams {a.cam_id}-{b.cam_id}: board corner consistency {cons[(a.cam_id, b.cam_id)] * 1000:.1f} mm "
                         f"> {max_consistency_m * 1000:.0f} mm (check square_length_m and intrinsics)")
    return ExtrinsicsResult(cams, cons, warns)
