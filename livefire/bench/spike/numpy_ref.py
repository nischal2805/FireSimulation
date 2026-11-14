"""NumPy reference for the two spike kernels (float32 throughout).

Conventions (shared by the Warp and Taichi kernels):
  * Advection works in grid-index space. Cell (i, j, k) is centred at index coordinate (i, j, k); array axes
    (0, 1, 2) correspond to world (x, y, z); velocity component c is along axis c. Velocity is in m/s, `h` in
    metres, `dt` in seconds, so a displacement in cells is `u * dt / h`.
  * Out-of-range sample positions are clamped to the grid (clamp-to-edge), then trilinearly interpolated.
  * Arrays are host NumPy arrays; the functions return new arrays and never modify their inputs.
"""

from __future__ import annotations

import numpy as np


def _trilinear(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Trilinear sample of `f` (G,G,G[,C]) at index coordinates `p` (...,3), clamp-to-edge."""
    g = f.shape[0]
    p = np.clip(p, 0.0, np.float32(g - 1)).astype(np.float32)
    i0 = np.minimum(np.floor(p).astype(np.int64), g - 2)
    t = (p - i0).astype(np.float32)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (t[..., 0] if dx else np.float32(1) - t[..., 0])
                    * (t[..., 1] if dy else np.float32(1) - t[..., 1])
                    * (t[..., 2] if dz else np.float32(1) - t[..., 2])
                )
                v = f[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
                out = out + (w[..., None] * v if v.ndim > w.ndim else w * v)
    return np.asarray(out, dtype=np.float32)


def advect(scalar: np.ndarray, vel: np.ndarray, dt: float, h: float) -> np.ndarray:
    """Semi-Lagrangian advection with RK2 (midpoint) backtrace.

    scalar: (G,G,G) f32; vel: (G,G,G,3) f32 cell-centred, m/s; dt s; h m. Returns (G,G,G) f32.
    """
    g = scalar.shape[0]
    c = np.float32(dt) / np.float32(h)
    idx = np.stack(np.meshgrid(*[np.arange(g, dtype=np.float32)] * 3, indexing="ij"), axis=-1)
    x_mid = idx - np.float32(0.5) * c * vel
    u_mid = _trilinear(vel, x_mid)
    x_back = idx - c * u_mid
    return _trilinear(scalar, x_back)


def conduction(
    temp: np.ndarray, cap: np.ndarray, neighbors: np.ndarray, gij: np.ndarray, dt: float
) -> np.ndarray:
    """`T_new_i = T_i + dt * sum_j G_ij (T_j - T_i) / Cap_i`; neighbours -1 padded.

    temp (N,) K; cap (N,) J/K; neighbors (N,K) i32; gij (N,K) W/K; dt s. Returns (N,) f32.
    """
    valid = neighbors >= 0
    tj = temp[np.where(valid, neighbors, 0)]
    flux = np.where(valid, gij * (tj - temp[:, None]), np.float32(0)).astype(np.float32)
    acc = flux.sum(axis=1, dtype=np.float32)
    return (temp + np.float32(dt) * acc / cap).astype(np.float32)
