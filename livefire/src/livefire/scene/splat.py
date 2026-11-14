"""Per-view occlusion depth by z-buffered disc splatting of surface elements (PRD 6.1)."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.common.geometry import AABB, CameraCalibration, project, ray_dirs_camera

DISC_FACTOR = 1.25
_FRINGE = 15  # px, window for silhouette-fringe detection (splat discs reach several px past the silhouette)


def iter_splat_pixels(
    calib: CameraCalibration, pos: NDArray, area: NDArray, normal: NDArray | None = None
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]]:
    """Yield ``(element_idx, flat_pixel, z)`` candidate triples (1-D, equal length), one batch per
    integer pixel radius. ``flat_pixel = v * W + u``; ``z`` is the element's Z-depth at that pixel
    (centre Z, or the tangent-plane depth clamped to +-radius when ``normal`` is given).
    Each element is a camera-facing disc of world radius ``DISC_FACTOR sqrt(area/pi)``.
    Shared by the z-buffer here and by the renderer's char/removed overlay."""
    h, w = calib.height, calib.width
    uv, z, ok = project(calib, pos)
    r_w = DISC_FACTOR * np.sqrt(np.asarray(area, dtype=np.float64) / np.pi)
    r_px = r_w * calib.K[0, 0] / np.where(ok, z, 1.0)
    if normal is not None:
        pc = (np.atleast_2d(pos) - calib.center_world) @ calib.T_world_from_camera[:3, :3]
        nc = np.asarray(normal, dtype=np.float64) @ calib.T_world_from_camera[:3, :3]
        npc = np.einsum("ij,ij->i", nc, pc)
        fx, fy, cx, cy = calib.K[0, 0], calib.K[1, 1], calib.K[0, 2], calib.K[1, 2]
    k = np.ceil(r_px).astype(np.int64)
    for kk in np.unique(k[ok]):
        sel = np.flatnonzero(ok & (k == kk))
        off = np.arange(-kk, kk + 1)
        du, dv = np.meshgrid(off, off)
        du, dv = du.ravel(), dv.ravel()
        # candidate pixel = round(uv) + offset; coverage by distance of pixel centre to disc centre
        u0 = np.rint(uv[sel, 0]).astype(np.int64)
        v0 = np.rint(uv[sel, 1]).astype(np.int64)
        pu = u0[:, None] + du[None, :]
        pv = v0[:, None] + dv[None, :]
        d2 = (pu - uv[sel, 0:1]) ** 2 + (pv - uv[sel, 1:2]) ** 2
        m = (d2 <= r_px[sel, None] ** 2) & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
        flat = (pv * w + pu)[m]
        zz = np.broadcast_to(z[sel, None], m.shape)[m]
        if normal is not None:
            den = nc[sel, 0:1] * (pu - cx) / fx + nc[sel, 1:2] * (pv - cy) / fy + nc[sel, 2:3]
            zp = npc[sel, None] / np.where(np.abs(den) < 1e-6, 1e-6, den)
            zp = np.clip(zp, z[sel, None] - r_w[sel, None], z[sel, None] + r_w[sel, None])
            zz = zp[m]
        eidx = np.broadcast_to(sel[:, None], m.shape)[m]
        yield eidx, flat, zz


def splat_depth(
    calib: CameraCalibration, pos: NDArray, area: NDArray, normal: NDArray | None = None
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Z-buffer (H,W) float32 (inf where uncovered) and coverage mask (H,W).

    Each element is a camera-facing disc of world radius ``1.25 sqrt(area/pi)``; its depth is the
    element centre Z-depth. If ``normal`` is given, the depth varies across the disc along the
    element's tangent plane (ray/plane intersection, clamped to +-radius of the centre Z) so the
    z-buffer is not biased towards the near edge of tilted surfaces. Elements are grouped by integer pixel radius and splatted
    with vectorised candidate generation (no per-pixel Python loops)."""
    h, w = calib.height, calib.width
    zbuf = np.full(h * w, np.inf)
    for _, flat, zz in iter_splat_pixels(calib, pos, area, normal):
        # min-reduce per pixel: sort by (pixel, z), keep first of each pixel
        o = np.lexsort((zz, flat))
        flat, zz = flat[o], zz[o]
        first = np.ones(len(flat), dtype=bool)
        first[1:] = flat[1:] != flat[:-1]
        f, zf = flat[first], zz[first]
        zbuf[f] = np.minimum(zbuf[f], zf)
    covered = np.isfinite(zbuf).reshape(h, w)
    return zbuf.reshape(h, w).astype(np.float32), covered


def known_clear(
    calib: CameraCalibration, measured: NDArray, valid: NDArray, roi: AABB
) -> NDArray[np.bool_]:
    """Pixels whose measured Z-depth is valid and beyond the ROI exit along the ray (or whose
    ray misses the ROI): nothing inside the ROI can occlude them."""
    h, w = measured.shape
    dc = ray_dirs_camera(calib).reshape(-1, 3)  # z component 1 -> t is Z-depth
    dw = dc @ calib.T_world_from_camera[:3, :3].T
    o = np.broadcast_to(calib.center_world, dw.shape)
    _, t_far, hit = roi.ray_intersect(o, dw)
    beyond = np.asarray(measured, dtype=np.float64).ravel() > t_far
    return (np.asarray(valid, dtype=bool).ravel() & (beyond | ~hit)).reshape(h, w)


def view_occlusion(
    calib: CameraCalibration,
    pos: NDArray,
    area: NDArray,
    measured: NDArray,
    valid: NDArray,
    roi: AABB,
    normal: NDArray | None = None,
    spacing: float = 0.0,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """``(view_depth, view_hole_mask)``: element splat; else known-clear measured depth; else a
    hole (depth 0, mask True).

    Silhouette rule: where the measured depth is valid and nearer than the chosen depth by more than
    ``max(0.5 * spacing, 0.5 % of the depth)``, the measured depth wins (``min(splat, measured)``). At silhouette pixels
    no element splat of the front object may cover the pixel, so the splat falls back to a surface behind it; the
    measured (frozen) depth is evidence of a nearer surface, so it is used (conservative for occlusion). Such
    pixels may have no element within 2 spacings (a click there reports "no surface")."""
    zbuf, covered = splat_depth(calib, pos, area, normal)
    clear = known_clear(calib, measured, valid, roi) & ~covered
    depth = np.where(covered, zbuf, np.where(clear, np.asarray(measured, dtype=np.float32), 0.0))
    hole = ~covered & ~clear
    m = np.asarray(measured, dtype=np.float64)
    tol = np.maximum(0.5 * spacing, 0.005 * depth)  # NOTE: tighter than max(2 spacings, 2 %): that left 3 cm errors
    ok = np.asarray(valid, dtype=bool) & (m > 0) & (covered | clear)
    nearer = ok & (m + tol < depth)
    # Fringe rule: element splats are dilated discs (up to ~2 px past the true silhouette), so at a silhouette the
    # splat of the front object also covers pixels that truly see the farther surface. There (and only there: a
    # splat pixel with a farther or uncovered pixel within the fringe window) a measured depth farther than the splat by more
    # than ``tol`` wins too, otherwise a click just outside the silhouette would resolve to the front object.
    d_far = np.where(covered, depth, 1e6).astype(np.float32)
    edge = cv2.dilate(d_far, np.ones((_FRINGE, _FRINGE), np.uint8)) > depth + 0.5 * spacing
    farther = ok & covered & edge & (m > depth + tol)
    depth = np.where(nearer | farther, m, depth)
    return depth.astype(np.float32), hole
