"""M2 latency statistics from the live per-frame CSV (P1-007, PRD 5.5 and 11.1 M2).

Input: ``out/live/live_<time>/latency.csv`` written by live mode, one row per camera per presented frame with the
columns ``frame, cam, seq, state, t_arrive_ns, t_use_ns, t_present_ns, frame_age_ms, capture_to_display_ms,
composite_ms, frame_ms, dropped_frame, sim_lag_s, fps, cams_shown``.

M2 = ``capture_to_display_ms = t_present - t_arrive`` (host clock, ``perf_counter_ns``), target 100 ms or less. The
verdict gates on the **median** (p50 <= target; controller ruling); p95/p99/max are informational. Steady state only: rows
whose ``state`` is in ``--states`` (default SIMULATING, PAUSED) and after the first ``--skip-frames`` presented frames
(default 30: warm-up, GPU clocks, first renders). Frame-level numbers (frame-time p50/p95/p99, delivered fps from the
``t_present_ns`` stamps, dropped-frame rate, simulation lag) use one entry per ``frame`` number, not one per camera.

Glass-to-glass (PRD 5.5) is a separate, one-time **camera-side offset** per camera measured with ``latency-timer`` and
``capture --show-timer`` (docs/RUNBOOK_P1.md section 9). Enter your readings with ``--glass CAM=ms[,ms...]`` (repeatable);
the per-camera median goes into ``g2g_offset_ms`` and is reported separately: it is never added into M2. The
convenience column ``c2d_p50_plus_offset_ms`` is the sum of that camera's p50 and its offset, labelled as an estimate.

Output: ``eval/results/latency_<label>.csv``, one row per scope (``all``, ``cam0``, ...).
Usage: ``uv run python eval/latency.py out/live/live_<time>/latency.csv [--label rtx4060_dual] [--glass 0=110,115 --glass 1=140]``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from livefire.common.eval_csv import percentile, read_csv, write_csv

LIVE_COLUMNS = (
    "frame", "cam", "seq", "state", "t_arrive_ns", "t_use_ns", "t_present_ns", "frame_age_ms",
    "capture_to_display_ms", "composite_ms", "frame_ms", "dropped_frame", "sim_lag_s", "fps", "cams_shown",
)  # fmt: skip
FIELDS = (
    "scope", "n_rows", "n_frames", "c2d_mean_ms", "c2d_p50_ms", "c2d_p95_ms", "c2d_p99_ms", "c2d_max_ms",
    "age_p50_ms", "age_p95_ms", "composite_p50_ms", "composite_p95_ms", "frame_p50_ms", "frame_p95_ms",
    "frame_p99_ms", "fps_delivered", "dropped_rate", "sim_lag_max_s", "sim_lag_end_s", "target_ms", "meets_target_p50",
    "g2g_offset_ms", "g2g_n", "c2d_p50_plus_offset_ms",
)  # fmt: skip
DEFAULT_STATES = ("SIMULATING", "PAUSED")
TARGET_MS = 100.0


@dataclass(frozen=True)
class LatencyRows:
    """Parsed numeric columns of a live latency CSV (one entry per row, file order)."""

    frame: np.ndarray
    cam: np.ndarray
    state: np.ndarray
    t_present_ns: np.ndarray
    cols: dict[str, np.ndarray]


def load_latency_csv(path: str | Path) -> LatencyRows:
    """Parse ``latency.csv``. Raises ``ValueError`` naming the missing columns / empty file."""
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = [c for c in LIVE_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (expected the live-mode latency.csv)")
    num = {c: np.array([float(r[c]) for r in rows]) for c in LIVE_COLUMNS if c != "state"}
    return LatencyRows(num["frame"].astype(int), num["cam"].astype(int), np.array([r["state"] for r in rows]),
                       num["t_present_ns"].astype(np.int64), num)  # fmt: skip


def parse_glass(items: list[str]) -> dict[int, list[float]]:
    """``["0=110,115", "1=140"]`` -> ``{0: [110.0, 115.0], 1: [140.0]}``. Raises ``ValueError`` on bad input."""
    out: dict[int, list[float]] = {}
    for s in items:
        try:
            cam, vals = s.split("=", 1)
            out.setdefault(int(cam), []).extend(float(v) for v in vals.split(",") if v.strip())
        except ValueError:
            raise ValueError(f"--glass expects CAM=ms[,ms...] (e.g. 0=110,115), got {s!r}") from None
    for c, v in out.items():
        if not v or any(x < 0 or not np.isfinite(x) for x in v):
            raise ValueError(f"--glass cam {c}: need at least one non-negative number")
    return out


def m2_statistics(rows: LatencyRows, *, states: tuple[str, ...] = DEFAULT_STATES, skip_frames: int = 30,
                  target_ms: float = TARGET_MS, glass: dict[int, list[float]] | None = None) -> list[dict[str, Any]]:
    """M2 summary rows (``FIELDS``): scope ``all`` (every camera row) then ``cam<c>``. Raises ``ValueError`` if no row
    survives the state / warm-up filter."""
    first = int(rows.frame.min())
    keep = np.isin(rows.state, list(states)) & (rows.frame >= first + skip_frames)
    if not keep.any():
        raise ValueError(f"no rows with state in {states} after skipping {skip_frames} frames "
                         f"(states present: {sorted(set(rows.state.tolist()))}, frames {first}..{int(rows.frame.max())})")  # fmt: skip
    glass = glass or {}
    out = []
    for scope, sel in [("all", keep)] + [(f"cam{c}", keep & (rows.cam == c)) for c in sorted(set(rows.cam[keep].tolist()))]:
        c2d, age = rows.cols["capture_to_display_ms"][sel], rows.cols["frame_age_ms"][sel]
        comp = rows.cols["composite_ms"][sel]
        # frame-level: one entry per frame number (a frame appears once per camera shown)
        fr, idx = np.unique(rows.frame[sel], return_index=True)
        ft, tp = rows.cols["frame_ms"][sel][idx], rows.t_present_ns[sel][idx]
        order = np.argsort(tp)
        span = (tp[order[-1]] - tp[order[0]]) / 1e9 if len(fr) > 1 else 0.0
        # frame_ms of the first presented frame of the run is 0 by construction: never counted
        ft_ok = ft[ft > 0]
        dropped = rows.cols["dropped_frame"][sel][idx]
        lag = rows.cols["sim_lag_s"][sel][idx]
        row_p50 = percentile(c2d, 50)
        row: dict[str, Any] = {
            "scope": scope, "n_rows": int(sel.sum()), "n_frames": len(fr), "c2d_mean_ms": float(c2d.mean()),
            "c2d_p50_ms": percentile(c2d, 50), "c2d_p95_ms": percentile(c2d, 95), "c2d_p99_ms": percentile(c2d, 99),
            "c2d_max_ms": float(c2d.max()), "age_p50_ms": percentile(age, 50), "age_p95_ms": percentile(age, 95),
            "composite_p50_ms": percentile(comp, 50), "composite_p95_ms": percentile(comp, 95),
            "frame_p50_ms": percentile(ft_ok, 50), "frame_p95_ms": percentile(ft_ok, 95), "frame_p99_ms": percentile(ft_ok, 99),
            "fps_delivered": (len(fr) - 1) / span if span > 0 else float("nan"),
            "dropped_rate": float(dropped.sum() / max(len(fr), 1)), "sim_lag_max_s": float(lag.max()),
            "sim_lag_end_s": float(lag[np.argmax(fr)]), "target_ms": target_ms, "meets_target_p50": row_p50 <= target_ms,
            "g2g_offset_ms": "", "g2g_n": "", "c2d_p50_plus_offset_ms": "",
        }  # fmt: skip
        if scope.startswith("cam") and int(scope[3:]) in glass:
            g = glass[int(scope[3:])]
            row.update(g2g_offset_ms=float(np.median(g)), g2g_n=len(g), c2d_p50_plus_offset_ms=row["c2d_p50_ms"] + float(np.median(g)))
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="latency.csv from out/live/live_<time>/")
    ap.add_argument("--label", default=None, help="name for the output file (default: the run folder name)")
    ap.add_argument("--states", default=",".join(DEFAULT_STATES), help="comma-separated lifecycle states to include")
    ap.add_argument("--skip-frames", type=int, default=30, help="warm-up frames to ignore")
    ap.add_argument("--target-ms", type=float, default=TARGET_MS)
    ap.add_argument("--glass", action="append", default=[], metavar="CAM=ms[,ms...]",
                    help="glass-to-glass readings from latency-timer for one camera (repeatable)")
    ap.add_argument("--out-dir", default="eval/results")
    a = ap.parse_args(argv)
    try:
        rows = load_latency_csv(a.csv)
        stats = m2_statistics(rows, states=tuple(s for s in a.states.split(",") if s), skip_frames=a.skip_frames,
                              target_ms=a.target_ms, glass=parse_glass(a.glass))
    except (ValueError, OSError) as e:
        print(f"latency eval failed: {e}", file=sys.stderr)
        return 2
    label = a.label or Path(a.csv).resolve().parent.name
    path = write_csv(Path(a.out_dir) / f"latency_{label}.csv", FIELDS, stats)
    for r in stats:
        print(f"{r['scope']:5s} n={r['n_rows']:5d}  capture-to-display p50 {r['c2d_p50_ms']:.1f} / p95 {r['c2d_p95_ms']:.1f} / "
              f"p99 {r['c2d_p99_ms']:.1f} ms  fps {r['fps_delivered']:.1f}  dropped {r['dropped_rate']:.1%}  "
              f"M2 target {a.target_ms:.0f} ms (median): {'MET' if r['meets_target_p50'] else 'NOT MET'}")  # fmt: skip
        if r["g2g_offset_ms"] != "":
            print(f"      glass-to-glass camera offset (separate, not in M2): median {r['g2g_offset_ms']:.0f} ms of {r['g2g_n']} readings")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
