"""Click picking on the frozen snapshot (P1R5): pixel -> world point and nearest object.

Frames: pixels are full-resolution ``(u, v)`` of the camera view (pixel centres at integer coordinates, OpenCV);
world points are metres in the table frame (+Z up). Host NumPy only; the snapshot is read-only and never modified.
Clicks use the frozen ``snapshot.view_depth`` (never live depth)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livefire.common.geometry import backproject
from livefire.scene.snapshot import SceneSnapshot

_PICK_SPACINGS = 2.0  # pick radius in surfel spacings
_PLANE_TOL = 0.25  # max normal distance of the point from an element plane, in spacings
_AMBIG_M = 0.01  # m, normal-distance below which two objects' surfaces are indistinguishable at a pick
_INPLANE_W = 0.05  # weight of the in-plane distance relative to the normal distance in the pick score


@dataclass(frozen=True)
class ClickPoint:
    """Resolved click: ``point`` world (3,) m or None; ``reason`` ``ok|outside|hole|no_depth``; ``pixel`` the integer
    pixel ``(u, v)`` whose depth neighbourhood was used (the nearest pixel to the click); ``depth`` the Z-depth [m]
    used; ``n_used`` how many valid neighbourhood pixels contributed."""

    point: np.ndarray | None
    reason: str
    pixel: tuple[int, int] | None = None
    depth: float = 0.0
    n_used: int = 0


def resolve_click(snapshot: SceneSnapshot, cam_id: int, u: float, v: float) -> ClickPoint:
    """Back-project the click ``(u, v)`` [px, full resolution] of camera ``cam_id`` using the frozen depth.

    A hole pixel gives ``hole`` (depth is never borrowed from neighbours, which may be another surface). Reference
    depth: the clicked pixel's own depth. The depth used is the median of the valid 3x3 neighbours within ``max(2 surfel spacings, 2 % of the
    reference depth)`` of the reference, so a click near a silhouette never takes the depth of the surface behind.
    Reasons: ``outside`` (outside the image), ``hole``, ``no_depth`` (unknown camera)."""
    if cam_id not in snapshot.view_depth:
        return ClickPoint(None, "no_depth")
    depth, hole = snapshot.view_depth[cam_id], snapshot.view_hole_mask[cam_id]
    h, w = depth.shape
    if not (np.isfinite(u) and np.isfinite(v)):
        return ClickPoint(None, "outside")
    ui, vi = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
    if not (0 <= ui < w and 0 <= vi < h):
        return ClickPoint(None, "outside")
    if hole[vi, ui] or not depth[vi, ui] > 0:
        return ClickPoint(None, "hole", (ui, vi))  # never borrow depth from neighbours: it may be another surface
    sl = (slice(max(vi - 1, 0), vi + 2), slice(max(ui - 1, 0), ui + 2))
    d = depth[sl].astype(np.float64).ravel()
    ok = (~hole[sl] & (depth[sl] > 0)).ravel()
    ref = float(depth[vi, ui])
    tol = max(2 * snapshot.surfel_spacing, 0.02 * ref)
    keep = ok & (np.abs(d - ref) <= tol)
    z = float(np.median(d[keep]))
    p = backproject(snapshot.cameras[cam_id], np.array([[float(u), float(v)]]), np.array([z]))[0]
    return ClickPoint(p, "ok", (ui, vi), z, int(keep.sum()))


def pixel_to_world(snapshot: SceneSnapshot, cam_id: int, u: float, v: float) -> tuple[np.ndarray | None, str]:
    """``(point_world (3,) float64 m | None, reason)`` of :func:`resolve_click`; reason ``"ok"`` or
    ``outside | hole | no_depth``."""
    r = resolve_click(snapshot, cam_id, u, v)
    return r.point, r.reason


def pick_object(snapshot: SceneSnapshot, cam_id: int, u: float, v: float) -> int:
    """Object id of the surface element nearest to the back-projected click, if within 2 surfel spacings;
    ``-1`` for holes, empty space or when no element is that close (also for elements with unknown id -1)."""
    p, _ = pixel_to_world(snapshot, cam_id, u, v)
    if p is None:
        return -1
    return pick_object_at(snapshot, p)


def pick_object_at(snapshot: SceneSnapshot, point: np.ndarray) -> int:
    """Like :func:`pick_object` for a world point (3,) m."""
    return pick_element_at(snapshot, point)[1]


def pick_element_at(snapshot: SceneSnapshot, point: np.ndarray) -> tuple[int, int]:
    """``(element index | -1, obj_id | -1)`` of the surfel that best explains world ``point`` among those within
    2 surfel spacings. Score = squared distance along the element normal + 0.05 x squared in-plane distance, so at
    contact lines the surface the point actually lies on wins over a perpendicular neighbour that is merely close."""
    if snapshot.pos.shape[0] == 0:
        return -1, -1
    d = snapshot.pos.astype(np.float64) - np.asarray(point, np.float64)
    d2 = np.sum(d * d, axis=1)
    dn = np.sum(d * snapshot.normal.astype(np.float64), axis=1)
    # within 2 spacings and within 0.25 spacing of the element's own plane (an element the point is far above is not the
    # surface that was hit, e.g. a table element under the side of a box)
    cand = (d2 <= (_PICK_SPACINGS * snapshot.surfel_spacing) ** 2) & (np.abs(dn) <= _PLANE_TOL * snapshot.surfel_spacing)
    if not cand.any():
        return -1, -1
    score = np.where(cand, dn * dn + _INPLANE_W * np.maximum(d2 - dn * dn, 0.0), np.inf)
    i = int(np.argmin(score))
    # Ambiguity: another object has an element in range that is (nearly) as well aligned with the point, e.g. cloth
    # lying 3 mm above the table at its rim. Surfel resolution cannot tell them apart, so report no pick (-1).
    best_dn = abs(float(dn[i]))
    other = (d2 <= (_PICK_SPACINGS * snapshot.surfel_spacing) ** 2) & (snapshot.obj_id != snapshot.obj_id[i])
    if other.any() and float(np.min(np.abs(dn[other]))) <= max(2.0 * best_dn, _AMBIG_M):
        return -1, -1
    return i, int(snapshot.obj_id[i])
