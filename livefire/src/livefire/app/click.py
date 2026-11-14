"""Click picking on the frozen snapshot (P1R5): pixel -> world point and nearest object.

Frames: pixels are full-resolution ``(u, v)`` of the camera view (pixel centres at integer coordinates, OpenCV);
world points are metres in the table frame (+Z up). Host NumPy only; the snapshot is read-only and never modified.
Clicks use the frozen ``snapshot.view_depth`` (never live depth)."""

from __future__ import annotations

import numpy as np

from livefire.common.geometry import backproject
from livefire.scene.snapshot import SceneSnapshot

_PICK_SPACINGS = 2.0  # pick radius in surfel spacings


def pixel_to_world(snapshot: SceneSnapshot, cam_id: int, u: float, v: float) -> tuple[np.ndarray | None, str]:
    """Back-project the click ``(u, v)`` [px, full resolution] of camera ``cam_id``.

    Depth is the median of the valid (non-hole, > 0) Z-depths [m] in the 3x3 neighbourhood of the nearest pixel,
    which is robust at object edges. Returns ``(point_world (3,) float64 m | None, reason)`` where reason is
    ``"ok"`` or, with a ``None`` point: ``"outside"`` (outside the image), ``"hole"`` (the clicked pixel has no
    valid depth, e.g. the room walls) or ``"no_depth"`` (unknown camera or no valid depth around the pixel)."""
    if cam_id not in snapshot.view_depth:
        return None, "no_depth"
    depth, hole = snapshot.view_depth[cam_id], snapshot.view_hole_mask[cam_id]
    h, w = depth.shape
    ui, vi = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
    if not (np.isfinite(u) and np.isfinite(v)) or not (0 <= ui < w and 0 <= vi < h):
        return None, "outside"
    if hole[vi, ui] or not depth[vi, ui] > 0:
        return None, "hole"
    d = depth[max(vi - 1, 0) : vi + 2, max(ui - 1, 0) : ui + 2].astype(np.float64)
    ok = ~hole[max(vi - 1, 0) : vi + 2, max(ui - 1, 0) : ui + 2] & (d > 0)
    if not ok.any():
        return None, "no_depth"
    z = float(np.median(d[ok]))
    return backproject(snapshot.cameras[cam_id], np.array([[float(u), float(v)]]), np.array([z]))[0], "ok"


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
    """``(element index | -1, obj_id | -1)`` of the surfel nearest to world ``point`` within 2 surfel spacings."""
    if snapshot.pos.shape[0] == 0:
        return -1, -1
    d2 = np.sum((snapshot.pos.astype(np.float64) - np.asarray(point, np.float64)) ** 2, axis=1)
    i = int(np.argmin(d2))
    if d2[i] > (_PICK_SPACINGS * snapshot.surfel_spacing) ** 2:
        return -1, -1
    return i, int(snapshot.obj_id[i])
