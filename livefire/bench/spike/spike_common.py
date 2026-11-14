"""Shared spike helpers: seeded inputs, timing harness, environment info, local Appendix C CSV writer.

The CSV writer is local to the spike; the full BenchmarkRecorder comes in a later task.
"""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from livefire.common.timers import StageTimer

# PRD Appendix C, verbatim and in order.
COLUMNS = [
    "run_id", "date", "git_sha", "workload_version", "gpu", "driver", "cuda", "os", "python", "framework",
    "power_mode", "scene", "grid_g", "surface_elements", "views", "resolution", "mode", "degradation_level",
    "stage", "variant", "timing_method", "warmup_frames", "samples", "median_ms", "p95_ms", "p99_ms",
    "delivered_fps", "rt_factor", "peak_vram_alloc_mb", "peak_vram_reserved_mb", "notes",
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


def time_runner(
    run: Callable[[], None],
    sync: Callable[[], None],
    warmup: int = 5,
    repeats: int = 5,
    samples: int = 20,
) -> dict[str, float]:
    """Time `run` with device sync before and after each sample (perf_counter_ns+sync).

    The first warm-up launch (includes JIT/compile) is reported as `first_launch_ms`; all `warmup` launches
    are discarded. Returns median/p95/p99 over repeats*samples plus per-repeat medians.
    """
    t0 = time.perf_counter_ns()
    run()
    sync()
    first_ms = (time.perf_counter_ns() - t0) / 1e6
    for _ in range(warmup - 1):
        run()
    sync()
    overall = StageTimer()
    rep_med = []
    for _ in range(repeats):
        rep = StageTimer()
        for _ in range(samples):
            sync()
            t0 = time.perf_counter_ns()
            run()
            sync()
            ms = (time.perf_counter_ns() - t0) / 1e6
            rep.record("s", ms)
            overall.record("s", ms)
        rep_med.append(rep.summary()["s"]["median_ms"])
    s = overall.summary()["s"]
    return {
        "first_launch_ms": first_ms,
        "median_ms": s["median_ms"],
        "p95_ms": s["p95_ms"],
        "p99_ms": s["p99_ms"],
        "n": s["n"],
        "rep_median_min": min(rep_med),
        "rep_median_max": max(rep_med),
    }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_sha() -> str:
    """Short HEAD sha, suffixed `-dirty` if the working tree has changes."""
    sha = _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    return sha + ("-dirty" if _run(["git", "status", "--porcelain"]) else "")


def gpu_info(device: str) -> dict[str, str]:
    """gpu/driver/cuda/power_mode/clock strings.

    CPU: gpu/driver/cuda `none`, power_mode `unknown`. CUDA: nvidia-smi values (power_mode combines power
    management, power limit and pstate; `clocks` holds current and max SM clock), `unknown` where unavailable.
    """
    if device == "cpu":
        return {"gpu": "none", "driver": "none", "cuda": "none", "power_mode": "unknown", "clocks": ""}
    fields = [
        "name", "driver_version", "clocks.sm", "clocks.max.sm", "power.management", "power.limit", "pstate",
    ]  # fmt: skip
    q = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    parts = [p.strip() for p in q.splitlines()[0].split(",")] if q else []
    parts += ["unknown"] * (len(fields) - len(parts))
    v = dict(zip(fields, parts, strict=True))
    cuda = _run(["bash", "-c", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"]).replace("CUDA Version: ", "")
    return {
        "gpu": v["name"],
        "driver": v["driver_version"],
        "cuda": cuda or "unknown",
        "power_mode": f"power.management={v['power.management']}; power.limit={v['power.limit']}; pstate={v['pstate']}",
        "clocks": f"sm={v['clocks.sm']}; max_sm={v['clocks.max.sm']} (at run start; note boost throttling)",
    }


def env_info() -> dict[str, str]:
    """os and python strings for the CSV."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": ".".join(map(str, sys.version_info[:3])),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with exactly the Appendix C columns (missing keys become empty cells)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
