"""Tiny CSV helpers shared by the Gate 1 evaluation tools (``eval/*.py``, ``app/measure_cli.py``). pathlib only."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(path: str | Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ``rows`` (dicts) with columns ``fields`` to ``path`` (parents created, UTF-8, ``\\n`` line ends).
    Floats are written with ``repr`` precision so a CSV round-trips exactly. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})
    return p


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Rows of a CSV as dicts of strings."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else repr(float(v))
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    return v


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    """``np.percentile`` (linear interpolation) that returns NaN for an empty input."""
    a = np.asarray(values, dtype=np.float64)
    return float(np.percentile(a, q)) if a.size else float("nan")


def mae_report(error_cm: np.ndarray, tape_cm: np.ndarray) -> dict[str, float]:
    """M4 statistics from per-pair signed errors ``error_cm = recon - tape`` and the tape distances (both (n,), cm).
    Keys: n, mae_cm, median_abs_cm, p95_abs_cm, max_abs_cm, bias_cm, rmse_cm, mean_rel_err_pct."""
    e, t = np.asarray(error_cm, dtype=np.float64), np.asarray(tape_cm, dtype=np.float64)
    if e.size == 0 or e.shape != t.shape:
        raise ValueError("need at least one pair; error_cm and tape_cm must have the same shape")
    if np.any(t <= 0):
        raise ValueError("tape distances must be positive")
    a = np.abs(e)
    return {"n": float(e.size), "mae_cm": float(a.mean()), "median_abs_cm": float(np.median(a)),
            "p95_abs_cm": percentile(a, 95), "max_abs_cm": float(a.max()), "bias_cm": float(e.mean()),
            "rmse_cm": float(np.sqrt(np.mean(e * e))), "mean_rel_err_pct": float(100.0 * np.mean(a / t))}  # fmt: skip
