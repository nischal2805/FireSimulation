"""Extrinsics: per-camera board pose -> ``T_world_from_camera`` in the P1R2 world frame, with quality checks.

The board lies flat on the table, visible from every camera. Per camera ``solvePnP`` on the ChArUco corners (with the
fitted K/dist) gives ``T_camera_from_board``; ``T_world_from_camera = T_world_from_board @ T_board_from_camera``
(origin board centre, +X board X, +Z toward the cameras, table plane z = 0). Metres, host NumPy float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.calib.board import BoardSpec, T_world_from_board, board_corners_world, solve_board_pose
from livefire.calib.intrinsics import IntrinsicsResult, View
from livefire.common.geometry import invert_transform


@dataclass
class CameraExtrinsics:
    cam_id: int
    T_world_from_camera: NDArray[np.float64]
    reproj_rms_px: float
    n_corners: int
    flipped: bool = False


@dataclass
class ExtrinsicsResult:
    cameras: list[CameraExtrinsics]
    consistency_m: dict[tuple[int, int], float] = field(default_factory=dict)  # median triangulated-vs-layout dist
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"cam{c.cam_id}: reproj RMS {c.reproj_rms_px:.3f} px ({c.n_corners} corners), "
                 f"position {np.round(c.T_world_from_camera[:3, 3], 4).tolist()} m" for c in self.cameras]
        lines += [f"cams {a}-{b}: triangulated board corners vs layout, median {d * 1000:.2f} mm"
                  for (a, b), d in self.consistency_m.items()]
        return "\n".join(lines + [f"WARNING: {w}" for w in self.warnings])


def estimate_extrinsics(
    spec: BoardSpec,
    intrinsics: dict[int, IntrinsicsResult],
    views: dict[int, View],
    *,
    min_corners: int = 8,
    max_reproj_rms_px: float = 1.5,
    max_consistency_m: float = 0.005,
) -> ExtrinsicsResult:
    """World poses of every camera in ``views`` (cam_id -> detection in the distorted image of that camera).

    Raises ValueError when a camera sees fewer than ``min_corners`` corners or the cameras disagree about which
    side of the table they are on. Warnings (not errors) for reprojection or consistency above the limits."""
    T_wb = T_world_from_board(spec)
    cams: list[CameraExtrinsics] = []
    warns: list[str] = []
    for cid, v in sorted(views.items()):
        if len(v.ids) < min_corners:
            raise ValueError(f"cam{cid}: only {len(v.ids)} board corners visible (need >= {min_corners}); "
                             "move the board or the camera so the whole board is in view")
        it = intrinsics[cid]
        T_cb, rms = solve_board_pose(spec, v.corners, v.ids, it.K_raw, it.dist)
        cams.append(CameraExtrinsics(cid, T_wb @ invert_transform(T_cb), rms, len(v.ids)))
        if rms > max_reproj_rms_px:
            warns.append(f"cam{cid}: board reprojection RMS {rms:.2f} px > {max_reproj_rms_px} px")
    zs = np.array([c.T_world_from_camera[2, 3] for c in cams])
    if np.all(zs < 0):  # P1R2: cameras must be above the table; flip about world X (z -> -z, y -> -y)
        F = np.diag([1.0, -1.0, -1.0, 1.0])
        for c in cams:
            c.T_world_from_camera, c.flipped = F @ c.T_world_from_camera, True
    elif np.any(zs <= 0):
        raise ValueError(f"cameras on both sides of the table plane (z = {zs.round(3).tolist()}): board detection is inconsistent")

    cons: dict[tuple[int, int], float] = {}
    layout = board_corners_world(spec)
    for a, b in combinations(cams, 2):
        va, vb = views[a.cam_id], views[b.cam_id]
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
