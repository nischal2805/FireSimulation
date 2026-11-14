"""Shared spike helpers: seeded inputs; timing, environment info and the Appendix C CSV writer come from
`livefire.common.bench_csv`."""

from __future__ import annotations

import numpy as np

# Shared Appendix C writer, env/GPU info and sync timing live in livefire.common.bench_csv (re-exported here).
from livefire.common.bench_csv import COLUMNS, env_info, git_sha, gpu_info, time_runner, write_rows

__all__ = [
    "COLUMNS", "env_info", "git_sha", "gpu_info", "make_advect_inputs", "make_conduction_inputs", "time_runner",
    "write_rows",
]  # fmt: skip


def make_advect_inputs(g: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded scalar (G,G,G) f32 in [0,1) and velocity (G,G,G,3) f32 in [-2,2] m/s (cell-centred)."""
    rng = np.random.default_rng(seed)
    scalar = rng.random((g, g, g), dtype=np.float32)
    vel = (rng.random((g, g, g, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    return scalar, vel


def make_conduction_inputs(
    n: int, k: int, seed: int, window: int = 256, t_lo: float = 293.0, t_span: float = 500.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded (temp K, cap J/K, neighbors i32 (N,K) -1 padded, G W/K).

    Neighbours lie within +-`window` indices of i (spatially sorted surfel graph assumption), 8..K valid
    per row; temperatures are `t_lo + t_span * U[0,1)` K; the graph is not symmetric (the kernel does not require it). # MOCK: synthetic random graph.
    """
    rng = np.random.default_rng(seed)
    temp = (t_lo + t_span * rng.random(n, dtype=np.float32)).astype(np.float32)
    cap = (0.5 + 1.5 * rng.random(n, dtype=np.float32)).astype(np.float32)
    offs = rng.integers(-window, window + 1, size=(n, k))
    nb = np.clip(np.arange(n)[:, None] + offs, 0, n - 1).astype(np.int32)
    nvalid = rng.integers(min(8, k), k + 1, size=n)
    nb[np.arange(k)[None, :] >= nvalid[:, None]] = -1
    gij = (0.01 + 0.09 * rng.random((n, k), dtype=np.float32)).astype(np.float32)
    return temp, cap, nb, gij
