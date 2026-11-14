"""Surface elements from depth (PRD 6.1): normals, surfel binning, neighbour graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from livefire.common.geometry import AABB, CameraCalibration, backproject_depth

F64 = NDArray[np.float64]

DEPTH_JUMP_FRAC = 0.03
MIN_COS = 0.2


@dataclass(frozen=True)
class ViewPoints:
    """Per-view surface samples (flat arrays over accepted pixels)."""

    pos: F64  # (M,3) world points
    normal: F64  # (M,3) unit, oriented to the camera
    obj_id: NDArray[np.int32]  # (M,)
    footprint: F64  # (M,) pixel footprint area on the surface, m^2


def normals_from_depth(
    calib: CameraCalibration, depth: NDArray, valid: NDArray, obj_id: NDArray
) -> tuple[F64, F64, NDArray[np.bool_]]:
    """World points (H,W,3), unit normals (H,W,3) oriented towards the camera, ok mask (H,W).

    Normal = normalised cross of central differences (right-left, down-up). Pixels whose
    4-neighbourhood is invalid, crosses an ``obj_id`` change, or spans a depth jump
    ``> 0.03 * depth`` in either difference are not ok (normal 0)."""
    d = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    P = backproject_depth(calib, d, valid)
    h, w = d.shape
    ok = np.zeros((h, w), dtype=bool)
    c = (slice(1, h - 1), slice(1, w - 1))
    r, l_, dn, up = (slice(1, h - 1), slice(2, w)), (slice(1, h - 1), slice(0, w - 2)), \
        (slice(2, h), slice(1, w - 1)), (slice(0, h - 2), slice(1, w - 1))  # fmt: skip
    oid = np.asarray(obj_id)
    okc = valid[c] & valid[r] & valid[l_] & valid[dn] & valid[up]
    for nb in (r, l_, dn, up):
        okc &= oid[nb] == oid[c]
    okc &= np.abs(d[r] - d[l_]) <= DEPTH_JUMP_FRAC * d[c]
    okc &= np.abs(d[dn] - d[up]) <= DEPTH_JUMP_FRAC * d[c]
    dx = P[r] - P[l_]
    dy = P[dn] - P[up]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1)
    okc &= nn > 1e-12
    n = n / np.where(nn > 0, nn, 1.0)[..., None]
    to_cam = calib.center_world - P[c]
    flip = np.einsum("hwc,hwc->hw", n, to_cam) < 0
    n = np.where(flip[..., None], -n, n)
    N = np.zeros((h, w, 3))
    N[c] = np.where(okc[..., None], n, 0.0)
    ok[c] = okc
    return P, N, ok


def view_points(
    calib: CameraCalibration, depth: NDArray, valid: NDArray, obj_id: NDArray, roi: AABB
) -> ViewPoints:
    """Accepted pixels (ok normal, obj_id >= 0, inside the ROI) as world samples with footprints."""
    P, N, ok = normals_from_depth(calib, depth, valid, obj_id)
    m = ok & (np.asarray(obj_id) >= 0)
    p, n = P[m], N[m]
    m2 = roi.contains(p)
    p, n = p[m2], n[m2]
    oid = np.asarray(obj_id)[m][m2].astype(np.int32)
    z = np.asarray(depth, dtype=np.float64)[m][m2]
    v = calib.center_world - p
    cos = np.abs(np.einsum("ij,ij->i", n, v)) / np.linalg.norm(v, axis=1)
    fp = (z / calib.K[0, 0]) * (z / calib.K[1, 1]) / np.maximum(cos, MIN_COS)
    return ViewPoints(p, n, oid, fp)


@dataclass(frozen=True)
class Surfels:
    pos: NDArray[np.float32]
    normal: NDArray[np.float32]
    area: NDArray[np.float32]
    obj_id: NDArray[np.int32]


def bin_surfels(views: list[ViewPoints], roi: AABB, spacing: float, min_points: int = 3) -> Surfels:
    """Pool views into bins keyed ``(obj_id, floor((p - roi_min)/s))``. ``area`` is the max over
    views of the summed pixel footprints in the bin, capped at ``2 s^2``."""
    pos = np.concatenate([v.pos for v in views])
    nrm = np.concatenate([v.normal for v in views])
    oid = np.concatenate([v.obj_id for v in views])
    cell = np.floor((pos - roi.min) / spacing).astype(np.int64)
    base = int(np.ceil(roi.size.max() / spacing)) + 2
    key = ((oid.astype(np.int64) + 1) * base + cell[:, 0]) * base + cell[:, 1]
    key = key * base + cell[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    nb = len(uniq)
    out_pos = np.stack([np.bincount(inv, pos[:, k], nb) for k in range(3)], axis=1) / cnt[:, None]
    nsum = np.stack([np.bincount(inv, nrm[:, k], nb) for k in range(3)], axis=1)
    nl = np.linalg.norm(nsum, axis=1)
    out_n = nsum / np.where(nl > 0, nl, 1.0)[:, None]
    area = np.zeros(nb)
    off = 0
    for v in views:
        k = len(v.pos)
        area = np.maximum(area, np.bincount(inv[off : off + k], v.footprint, nb))
        off += k
    area = np.minimum(area, 2.0 * spacing**2)
    out_oid = np.zeros(nb, dtype=np.int32)
    out_oid[inv] = oid
    keep = (cnt >= min_points) & (nl > 1e-9)
    return Surfels(
        out_pos[keep].astype(np.float32),
        out_n[keep].astype(np.float32),
        area[keep].astype(np.float32),
        out_oid[keep],
    )


def build_neighbors(
    pos: NDArray, obj_id: NDArray, radius: float, k_max: int
) -> NDArray[np.int32]:
    """Symmetric neighbour lists (N,K), ``-1`` padded, nearest first. Radius search within the
    same object only (one KD-tree per object), self excluded, capped at ``k_max`` nearest, then
    an edge (i,j) is kept only if j in N(i) and i in N(j)."""
    n = len(pos)
    idx = np.full((n, k_max), -1, dtype=np.int64)
    p64 = np.asarray(pos, dtype=np.float64)
    for o in np.unique(obj_id):
        members = np.flatnonzero(obj_id == o)
        if len(members) < 2:
            continue
        tree = cKDTree(p64[members])
        k = min(k_max + 1, len(members))
        dist, j = tree.query(p64[members], k=k, distance_upper_bound=radius * (1 + 1e-9))
        j = np.where(np.isfinite(dist), j, -1)
        # drop self (column order: self is nearest, but guard against duplicates)
        self_col = j == np.arange(len(members))[:, None]
        j = np.where(self_col, -1, j)
        glob = np.where(j >= 0, members[np.clip(j, 0, None)], -1)
        order = np.argsort(glob < 0, axis=1, kind="stable")
        glob = np.take_along_axis(glob, order, axis=1)[:, :k_max]
        idx[members, : glob.shape[1]] = glob
    # symmetrise
    ii = np.repeat(np.arange(n), k_max)
    jj = idx.ravel()
    m = jj >= 0
    fwd = ii[m] * n + jj[m]
    rev = jj[m] * n + ii[m]
    keep = np.zeros(len(idx.ravel()), dtype=bool)
    keep[np.flatnonzero(m)] = np.isin(fwd, rev)
    idx = np.where(keep.reshape(idx.shape), idx, -1)
    order = np.argsort(idx < 0, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1).astype(np.int32)
