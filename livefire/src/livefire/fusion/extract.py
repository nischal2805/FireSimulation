"""Surface extraction from a TSDF: zero-crossing points with TSDF-gradient normals (PRD 6.1).

Frame: world, metres. Pure NumPy on host copies of the volume; outputs are caller-owned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from livefire.fusion.tsdf import TSDFGrid

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SurfacePoints:
    """Zero-crossing samples. ``pos`` (M,3) m world, ``normal`` (M,3) unit (TSDF gradient: points into free space),
    ``footprint`` (M,) m^2 = surface area one sample stands for (``h^2 / |n|_1``, exact on average for any plane
    orientation because the crossing density along the three axes is ``|n|_1 / h^2``)."""

    pos: F64
    normal: F64
    footprint: F64


def _gradient(t: NDArray, w: NDArray, h: float) -> NDArray[np.float64]:
    """Central-difference gradient (nx,ny,nz,3) of ``t`` using only observed neighbours (one-sided at the border of the
    observed region, 0 where no neighbour is observed)."""
    obs = w > 0
    g = np.zeros(t.shape + (3,))
    for a in range(3):
        n = t.shape[a]
        idx_p = np.minimum(np.arange(n) + 1, n - 1)
        idx_m = np.maximum(np.arange(n) - 1, 0)
        tp, tm = np.take(t, idx_p, axis=a), np.take(t, idx_m, axis=a)
        op, om = np.take(obs, idx_p, axis=a), np.take(obs, idx_m, axis=a)
        num = np.where(op, tp - t, 0.0) + np.where(om, t - tm, 0.0)
        den = op.astype(float) + om.astype(float)
        g[..., a] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0) / h
    return g


def extract_surface(tsdf: NDArray, weight: NDArray, grid: TSDFGrid, min_weight: float = 1e-6) -> SurfacePoints:
    """Zero crossings of ``tsdf`` between face-adjacent observed voxels (weight >= ``min_weight``), linearly
    interpolated along the voxel edge; normals from the TSDF gradient interpolated to the crossing and normalised."""
    t = np.asarray(tsdf, dtype=np.float64)
    wobs = np.asarray(weight) >= min_weight
    h = grid.h
    g = _gradient(t, wobs.astype(np.float32), h)
    pos, nrm = [], []
    for a in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[a], sl_b[a] = slice(0, -1), slice(1, None)
        sa, sb = tuple(sl_a), tuple(sl_b)
        ta, tb = t[sa], t[sb]
        m = wobs[sa] & wobs[sb] & ((ta > 0.0) != (tb > 0.0))
        i = np.argwhere(m)
        if not len(i):
            continue
        va = ta[m]
        vb = tb[m]
        f = va / (va - vb)  # fraction along the edge from voxel a to b
        pa = grid.centers(i)
        p = pa.copy()
        p[:, a] += f * h
        ib = i.copy()
        ib[:, a] += 1
        n = (1 - f)[:, None] * g[tuple(i.T)] + f[:, None] * g[tuple(ib.T)]
        pos.append(p)
        nrm.append(n)
    if not pos:
        return SurfacePoints(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    p, n = np.concatenate(pos), np.concatenate(nrm)
    nl = np.linalg.norm(n, axis=1)
    keep = nl > 1e-9
    p, n, nl = p[keep], n[keep] / nl[keep, None], nl[keep]
    fp = h * h / np.abs(n).sum(axis=1)
    return SurfacePoints(p, n, fp)
