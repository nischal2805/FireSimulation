"""Stage timing with perf_counter_ns. Times are milliseconds."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np


class StageTimer:
    """Collects per-stage wall-clock samples (ms). Not thread-safe."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, name: str, ms: float) -> None:
        """Add one sample of `ms` milliseconds to stage `name`."""
        self._samples.setdefault(name, []).append(float(ms))

    @contextmanager
    def stage(self, name: str, device_sync: Callable[[], None] | None = None) -> Iterator[None]:
        """Time the block. `device_sync` (if given) runs before each clock read (GPU boundary rule, PRD 5.5)."""
        if device_sync is not None:
            device_sync()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            if device_sync is not None:
                device_sync()
            self.record(name, (time.perf_counter_ns() - t0) / 1e6)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per stage: n, median_ms, p95_ms, p99_ms (numpy linear percentiles)."""
        out: dict[str, dict[str, float]] = {}
        for name, s in self._samples.items():
            a = np.asarray(s)
            out[name] = {
                "n": len(s),
                "median_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)),
            }
        return out

    def reset(self) -> None:
        """Drop all samples."""
        self._samples.clear()
