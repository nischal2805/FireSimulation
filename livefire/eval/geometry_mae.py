"""M4 geometry accuracy (P1-007, PRD 11.1 M4): metric MAE in cm against tape-measured distances, plus AbsRel and
completeness of the frozen depth on synthetic scenes.

**Real scene (the Gate 1 number).** ``python -m livefire.app measure`` lets you click two points in the frozen views and
type the tape distance; it saves ``pairs.csv`` (columns in ``app/measure_cli.py``). This tool reads that file::

    uv run python eval/geometry_mae.py --pairs out/eval/measure_<time>/pairs.csv --label desk_scene1

``error_cm = d_recon_cm - d_tape_cm`` per pair; **MAE = mean |error_cm|**; also median / p95 / max |error|, signed bias,
RMSE and mean relative error (% of the tape distance). Fewer than 10 pairs prints a warning (the MAE is then a rough
number); pairs should span short (5-10 cm) to long (50+ cm) distances, in both cameras and in all three directions.

**Synthetic depth quality.** ``--synthetic [--depth NAME]`` acquires the synthetic ``table_v1`` rig (oracle depth unless
``--depth`` names a model) and compares the frozen per-camera depth with the renderer's ground truth over pixels whose
true surface point lies in the fusion ROI: MAE [m], AbsRel = mean(|z - z_gt| / z_gt) over pixels valid in both, and
completeness = (valid in both) / (valid in ground truth). The same functions apply to any bundle + ground truth.

Output: ``eval/results/geometry_mae_<label>.csv`` (``metric, scope, value, unit, n``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from livefire.app.acq_bundle import AcqBundle, resolve_bundle
from livefire.common.eval_csv import percentile, read_csv, write_csv
from livefire.common.geometry import backproject_depth

FIELDS = ("metric", "scope", "value", "unit", "n", "note")
MIN_PAIRS = 10


def mae_report(error_cm: np.ndarray, tape_cm: np.ndarray) -> dict[str, float]:
    """M4 statistics from per-pair signed errors ``error_cm = recon - tape`` and the tape distances (both (n,), cm).
    Keys: n, mae_cm, median_abs_cm, p95_abs_cm, max_abs_cm, bias_cm, rmse_cm, mean_rel_err_pct."""
    e, t = np.asarray(error_cm, np.float64), np.asarray(tape_cm, np.float64)
    if e.size == 0 or e.shape != t.shape:
        raise ValueError("need at least one pair; error_cm and tape_cm must have the same shape")
    if np.any(t <= 0):
        raise ValueError("tape distances must be positive")
    a = np.abs(e)
    return {"n": float(e.size), "mae_cm": float(a.mean()), "median_abs_cm": float(np.median(a)),
            "p95_abs_cm": percentile(a, 95), "max_abs_cm": float(a.max()), "bias_cm": float(e.mean()),
            "rmse_cm": float(np.sqrt(np.mean(e * e))), "mean_rel_err_pct": float(100.0 * np.mean(a / t))}  # fmt: skip


def read_pairs(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(error_cm, tape_cm)`` from a ``pairs.csv`` (needs columns ``d_tape_cm``, ``d_recon_cm``)."""
    rows = read_csv(path)
    if not rows or "d_tape_cm" not in rows[0] or "d_recon_cm" not in rows[0]:
        raise ValueError(f"{path}: not a pairs.csv from `measure` (needs d_tape_cm and d_recon_cm columns)")
    tape = np.array([float(r["d_tape_cm"]) for r in rows])
    recon = np.array([float(r["d_recon_cm"]) for r in rows])
    return recon - tape, tape


def depth_metrics(depth: np.ndarray, hole: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray,
                  mask: np.ndarray | None = None) -> dict[str, float]:
    """Depth quality of one view against ground truth. ``depth`` f32 (H,W) m (0 = hole), ``hole`` bool (H,W), ``gt``
    (H,W) Z-depth m, ``gt_valid`` bool (H,W), ``mask`` optional bool (H,W) restricting the evaluated pixels (e.g. the ROI).
    Returns n_gt (evaluated GT-valid pixels), mae_m, absrel (both over pixels valid in both) and completeness."""
    sel = gt_valid if mask is None else (gt_valid & mask)
    both = sel & ~hole & (depth > 0)
    n_gt = int(sel.sum())
    if n_gt == 0:
        return {"n_gt": 0.0, "mae_m": float("nan"), "absrel": float("nan"), "completeness": float("nan")}
    if not both.any():
        return {"n_gt": float(n_gt), "mae_m": float("nan"), "absrel": float("nan"), "completeness": 0.0}
    e = np.abs(depth[both].astype(np.float64) - gt[both].astype(np.float64))
    return {"n_gt": float(n_gt), "mae_m": float(e.mean()), "absrel": float((e / gt[both]).mean()),
            "completeness": float(both.sum() / n_gt)}  # fmt: skip


def synthetic_depth_metrics(bundle: AcqBundle, truth: dict[int, Any]) -> dict[int, dict[str, float]]:
    """Per-camera ``depth_metrics`` of ``bundle`` against ``{cam: SyntheticView}`` ground truth, restricted to pixels
    whose true surface point lies inside ``bundle.roi`` (so room walls do not dominate completeness)."""
    out = {}
    for c, cal in bundle.cameras.items():
        tv = truth[c]
        pts = backproject_depth(cal, tv.depth, tv.valid).reshape(-1, 3)
        inroi = bundle.roi.contains(pts).reshape(tv.depth.shape) & tv.valid
        out[c] = depth_metrics(bundle.view_depth[c], bundle.view_hole[c], tv.depth, tv.valid, inroi)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=None, help="pairs.csv written by `python -m livefire.app measure`")
    ap.add_argument("--synthetic", action="store_true", help="AbsRel/completeness on the synthetic table_v1 rig")
    ap.add_argument("--bundle", default=None, help="with --synthetic: an acquisition.npz of that rig")
    ap.add_argument("--depth", default=None, help="with --synthetic: depth backend (default oracle)")
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--out", default="out/eval", help="with --synthetic: where the acquisition bundle is saved")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", default="eval/results")
    a = ap.parse_args(argv)
    if not a.pairs and not a.synthetic:
        ap.error("give --pairs pairs.csv (real scene) and/or --synthetic")
    rows: list[dict[str, Any]] = []
    label = a.label
    if a.pairs:
        try:
            err, tape = read_pairs(a.pairs)
            m = mae_report(err, tape)
        except (ValueError, OSError) as e:
            print(f"M4 failed: {e}", file=sys.stderr)
            return 2
        label = label or Path(a.pairs).resolve().parent.name
        n = int(m.pop("n"))
        units = {"mean_rel_err_pct": "%"}
        for k, v in m.items():
            rows.append({"metric": k, "scope": "tape_pairs", "value": v, "unit": units.get(k, "cm"), "n": n, "note": ""})
        print(f"M4 tape pairs: n={n}  MAE {m['mae_cm']:.2f} cm  median {m['median_abs_cm']:.2f}  p95 {m['p95_abs_cm']:.2f}  "
              f"max {m['max_abs_cm']:.2f}  bias {m['bias_cm']:+.2f} cm  mean relative error {m['mean_rel_err_pct']:.1f} %")
        if n < MIN_PAIRS:
            print(f"WARNING: only {n} pairs; measure at least {MIN_PAIRS} spread over short and long distances")
    if a.synthetic:
        from livefire.scene.synthetic import build_table_v1, render_view

        ns = argparse.Namespace(bundle=a.bundle, synthetic=True, depth=a.depth, grid=a.grid, out=a.out)
        bundle = resolve_bundle(ns)
        scene = build_table_v1()
        truth = {c: render_view(scene, cal) for c, cal in bundle.cameras.items()}
        label = label or "synthetic_" + (a.depth or "oracle")
        for c, m in synthetic_depth_metrics(bundle, truth).items():
            note = f"{bundle.label}; ROI pixels only"
            for k, unit in (("mae_m", "m"), ("absrel", "-"), ("completeness", "-")):
                rows.append({"metric": k, "scope": f"synthetic_cam{c}", "value": m[k], "unit": unit, "n": int(m["n_gt"]), "note": note})
            print(f"synthetic cam{c}: MAE {m['mae_m'] * 100:.2f} cm  AbsRel {m['absrel']:.4f}  completeness {m['completeness']:.3f} "
                  f"(n={int(m['n_gt'])} GT pixels in ROI)")  # fmt: skip
    path = write_csv(Path(a.out_dir) / f"geometry_mae_{label}.csv", FIELDS, rows)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
