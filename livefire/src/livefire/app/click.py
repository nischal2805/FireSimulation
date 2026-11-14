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

    Reference depth: the clicked pixel's own depth when valid; otherwise the nearest-depth cluster of the valid 3x3
    neighbours. The depth used is the median of the valid 3x3 neighbours within ``max(2 surfel spacings, 2 % of the
    reference depth)`` of the reference, so a click near a silhouette never takes the depth of the surface behind.
    Reasons: ``outside`` (outside the image), ``no_depth`` (unknown camera or no valid depth around the pixel)."""
    if cam_id not in snapshot.view_depth:
        return ClickPoint(None, "no_depth")
    depth, hole = snapshot.view_depth[cam_id], snapshot.view_hole_mask[cam_id]
    h, w = depth.shape
    if not (np.isfinite(u) and np.isfinite(v)):
        return ClickPoint(None, "outside")
    ui, vi = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
    if not (0 <= ui < w and 0 <= vi < h):
        return ClickPoint(None, "outside")
    sl = (slice(max(vi - 1, 0), vi + 2), slice(max(ui - 1, 0), ui + 2))
    d = depth[sl].astype(np.float64).ravel()
    ok = (~hole[sl] & (depth[sl] > 0)).ravel()
    if not ok.any():
        return ClickPoint(None, "hole" if hole[vi, ui] or not depth[vi, ui] > 0 else "no_depth", (ui, vi))
    centre_ok = not hole[vi, ui] and depth[vi, ui] > 0
    dv = np.sort(d[ok])
    if centre_ok:
        ref = float(depth[vi, ui])
    else:  # largest cluster of mutually close depths, nearest one on ties
        tol_c = np.maximum(2 * snapshot.surfel_spacing, 0.02 * dv)
        counts = [(np.abs(dv - x) <= t).sum() for x, t in zip(dv, tol_c)]
        ref = float(dv[int(np.argmax(counts))])
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
    cand = d2 <= (_PICK_SPACINGS * snapshot.surfel_spacing) ** 2
    if not cand.any():
        return -1, -1
    dn = np.sum(d * snapshot.normal.astype(np.float64), axis=1)
    score = np.where(cand, dn * dn + _INPLANE_W * np.maximum(d2 - dn * dn, 0.0), np.inf)
    i = int(np.argmin(score))
    return i, int(snapshot.obj_id[i])
