"""Benchmark recording: PRD Appendix C CSV schema, environment capture and device-synchronised timing.

``BenchmarkRecorder`` implements the ``livefire.common.interfaces.BenchmarkRecorder`` protocol. Rows hold the
Appendix C columns in order; environment fields (git sha, date, OS, python, framework, gpu/driver/cuda,
power mode) are filled automatically. CPU runs are labelled ``gpu=none`` and ``framework=warp-<ver>-cpu``
(PRD 11.4, global constraints). Times are milliseconds, memory is MB. The recorder owns its rows (host
memory) and is thread-safe.
"""

from __future__ import annotations

import csv
import datetime
import platform
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from livefire.common.timers import StageTimer

# PRD Appendix C, verbatim and in order.
COLUMNS = [
    "run_id", "date", "git_sha", "workload_version", "gpu", "driver", "cuda", "os", "python", "framework",
    "power_mode", "scene", "grid_g", "surface_elements", "views", "resolution", "mode", "degradation_level",
    "stage", "variant", "timing_method", "warmup_frames", "samples", "median_ms", "p95_ms", "p99_ms",
    "delivered_fps", "rt_factor", "peak_vram_alloc_mb", "peak_vram_reserved_mb", "notes",
]  # fmt: skip

# Columns that must be non-empty in every row (VRAM, fps, rt_factor and a few counts may be legitimately empty).
REQUIRED = [
    "run_id", "date", "git_sha", "workload_version", "gpu", "driver", "cuda", "os", "python", "framework",
    "power_mode", "scene", "mode", "degradation_level", "stage", "timing_method", "warmup_frames", "samples",
    "median_ms", "p95_ms", "p99_ms",
]  # fmt: skip

# value units accepted by ``record`` -> the column the value lands in
_UNIT_COLUMN = {
    "ms": "median_ms", "fps": "delivered_fps", "x": "rt_factor", "mb": "peak_vram_alloc_mb",
}  # fmt: skip


def time_runner(
    run: Callable[[], None],
    sync: Callable[[], None],
    warmup: int = 5,
    repeats: int = 5,
    samples: int = 20,
) -> dict[str, float]:
    """Time ``run`` with device sync before and after each sample (``perf_counter_ns`` + sync).

    The first warm-up launch (includes JIT/compile) is reported as ``first_launch_ms``; all ``warmup`` launches
    are discarded. Returns median/p95/p99 over ``repeats * samples`` timed samples (``n``) plus the smallest and
    largest per-repeat median.
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
    """Short HEAD sha, suffixed ``-dirty`` if the working tree has changes."""
    sha = _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    return sha + ("-dirty" if _run(["git", "status", "--porcelain"]) else "")


def gpu_info(device: str, power_mode: str | None = None) -> dict[str, str]:
    """gpu/driver/cuda/power_mode/clocks strings.

    CPU: gpu/driver/cuda ``none``, power_mode ``unknown`` (or ``power_mode`` if given). CUDA: nvidia-smi values
    (power_mode combines power management, power limit and pstate; ``clocks`` holds current and max SM clock),
    ``unknown`` where unavailable.
    """
    if device == "cpu":
        return {"gpu": "none", "driver": "none", "cuda": "none", "power_mode": power_mode or "unknown", "clocks": ""}
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
        "power_mode": power_mode
        or f"power.management={v['power.management']}; power.limit={v['power.limit']}; pstate={v['pstate']}",
        "clocks": f"sm={v['clocks.sm']}; max_sm={v['clocks.max.sm']} (at run start; note boost throttling)",
    }


def env_info() -> dict[str, str]:
    """``os`` and ``python`` strings for the CSV."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": ".".join(map(str, sys.version_info[:3])),
    }


def warp_framework(device: str) -> str:
    """``warp-<ver>`` (suffixed ``-cpu`` on the CPU device)."""
    import warp as wp

    return f"warp-{wp.__version__}" + ("-cpu" if device == "cpu" else "")


def vram_mb(device: str) -> tuple[float | str, float | str]:
    """``(peak allocated, peak reserved)`` device memory in MB from Warp's mempool high-water mark.

    Empty strings on CPU. Warp exposes no separate reserved counter: both values are the allocated high-water
    mark (process-cumulative)."""
    if device == "cpu":
        return "", ""
    import warp as wp

    mb = wp.get_mempool_used_mem_high(device) / 1e6
    return round(mb, 1), round(mb, 1)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with exactly the Appendix C columns (missing keys become empty cells)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


class BenchmarkRecorder:
    """Collects Appendix C rows and writes them to CSV. Implements the ``BenchmarkRecorder`` protocol.

    ``device`` is the framework device string (``"cpu"`` or ``"cuda:0"``); ``framework`` defaults to the installed
    Warp version. ``power_mode`` defaults to ``unknown`` on CPU and the nvidia-smi summary on GPU. ``base``
    overrides/extends the per-run fields (e.g. ``workload_version``, ``timing_method``, ``warmup_frames``).
    Environment fields are captured once at construction (git sha before any output is written).
    Thread-safe; rows live in host memory owned by the recorder.
    """

    def __init__(
        self,
        device: str = "cpu",
        *,
        framework: str | None = None,
        power_mode: str | None = None,
        run_id: str | None = None,
        workload_version: str | int = "",
        base: dict[str, Any] | None = None,
    ) -> None:
        info = gpu_info(device, power_mode)
        self.clocks = info.pop("clocks")
        self.device = device
        today = datetime.date.today().isoformat()
        self.base: dict[str, Any] = {
            "run_id": run_id or f"{'cpu' if device == 'cpu' else 'gpu'}-{today}-{uuid.uuid4().hex[:6]}",
            "date": today,
            "git_sha": git_sha(),
            "workload_version": workload_version,
            "framework": framework or warp_framework(device),
            **info,
            **env_info(),
            "timing_method": "perf_counter_ns+sync",
            "degradation_level": 0,
            **(base or {}),
        }
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Copy of the recorded rows."""
        with self._lock:
            return [dict(r) for r in self._rows]

    def add_row(self, **fields: Any) -> dict[str, Any]:
        """Append a full row: run fields from the constructor overlaid with ``fields`` (Appendix C column names).
        Raises ValueError for an unknown column."""
        bad = set(fields) - set(COLUMNS)
        if bad:
            raise ValueError(f"unknown Appendix C columns: {sorted(bad)}")
        row = {**self.base, **fields}
        with self._lock:
            self._rows.append(row)
        return row

    def record(self, name: str, value: float, unit: str, **labels: str | float) -> None:
        """Append one row for stage ``name``. ``unit`` selects the column ``value`` lands in: ``ms`` ->
        ``median_ms`` (and p95/p99 default to it, ``samples`` to 1), ``fps`` -> ``delivered_fps``, ``x`` ->
        ``rt_factor``, ``mb`` -> ``peak_vram_alloc_mb``. ``labels`` named like an Appendix C column set that
        column; any other label is appended to ``notes`` as ``key=value``. ValueError for an unknown unit."""
        if unit not in _UNIT_COLUMN:
            raise ValueError(f"unknown unit {unit!r}; expected one of {sorted(_UNIT_COLUMN)}")
        cols = {k: v for k, v in labels.items() if k in COLUMNS}
        extra = [f"{k}={v}" for k, v in labels.items() if k not in COLUMNS]
        fields: dict[str, Any] = {"stage": name, "samples": 1, "warmup_frames": 0, **cols}
        fields[_UNIT_COLUMN[unit]] = value
        if unit == "ms":
            fields.setdefault("p95_ms", value)
            fields.setdefault("p99_ms", value)
        if extra:
            fields["notes"] = "; ".join(([str(fields["notes"])] if "notes" in fields else []) + extra)
        self.add_row(**fields)

    def add_timing(self, stage: str, t: dict[str, float], *, warmup: int, **fields: Any) -> dict[str, Any]:
        """Append a row from a ``time_runner`` result ``t`` (median/p95/p99 in ms, ``samples`` = ``t['n']``)."""
        return self.add_row(
            stage=stage, warmup_frames=warmup, samples=int(t["n"]), median_ms=round(t["median_ms"], 4),
            p95_ms=round(t["p95_ms"], 4), p99_ms=round(t["p99_ms"], 4), **fields,
        )  # fmt: skip

    def flush(self, path: str) -> int:
        """Write all rows to ``path`` (overwrites), return the row count. Raises OSError on write failure."""
        with self._lock:
            rows = [dict(r) for r in self._rows]
        write_rows(Path(path), rows)
        return len(rows)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read an Appendix C CSV; raises ValueError if the header is not exactly the schema columns in order."""
    with Path(path).open(newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != COLUMNS:
            raise ValueError(f"{path}: header does not match Appendix C columns")
        return list(r)

