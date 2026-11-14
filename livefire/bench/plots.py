"""Benchmark plots (PRD 11.4: plots read only from CSVs).

Usage: python bench/plots.py [CSV ...] [--out DIR]
Default input: the newest bench/results/baseline_*.csv. Writes PNGs to bench/plots/:
  stage_scaling_<tag>.png  stage median ms (p95 whiskers) vs gas grid G, log-log
  render_frame_<tag>.png   render_view and frame_two_views median/p95 ms
Titles carry the gpu/framework labels from the CSV rows (CPU runs read `gpu=none, warp-<ver>-cpu`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from livefire.common.bench_csv import read_rows

HERE = Path(__file__).resolve().parent
SCALING_STAGES = ["sim_step", "gas_step", "projection", "surface_step"]


def _label(rows: list[dict[str, str]]) -> str:
    r = rows[0]
    return f"gpu={r['gpu']}, framework={r['framework']}, git {r['git_sha']}, workload v{r['workload_version']}"


def plot_scaling(rows: list[dict[str, str]], out: Path) -> Path:
    """Stage ms vs G (log-log) for the sim stages."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stage in SCALING_STAGES:
        pts = sorted((int(r["grid_g"]), float(r["median_ms"]), float(r["p95_ms"])) for r in rows if r["stage"] == stage)
        if not pts:
            continue
        g, med, p95 = zip(*pts, strict=True)
        ax.errorbar(g, med, yerr=[[0] * len(g), [b - a for a, b in zip(med, p95, strict=True)]], marker="o", capsize=3,
                    label=stage)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    gs = sorted({int(r["grid_g"]) for r in rows if r["stage"] in SCALING_STAGES})
    ax.set_xticks(gs, [str(g) for g in gs])
    ax.minorticks_off()
    ax.set_xlabel("gas grid G (cells per edge)")
    ax.set_ylabel("ms per call (median, whisker to p95)")
    ax.set_title(f"Stage cost vs grid size\n{_label(rows)}", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_render_frame(rows: list[dict[str, str]], out: Path) -> Path | None:
    """Median/p95 bars for render_view variants and frame_two_views."""
    sel = [r for r in rows if r["stage"] in ("render_view", "frame_two_views")]
    if not sel:
        return None
    names = [f"{r['stage']}\n{r['variant'].split(' ')[0]}" for r in sel]
    med = [float(r["median_ms"]) for r in sel]
    p95 = [float(r["p95_ms"]) for r in sel]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(sel))
    ax.bar([i - 0.2 for i in x], med, 0.4, label="median")
    ax.bar([i + 0.2 for i in x], p95, 0.4, label="p95")
    ax.set_xticks(list(x), names, fontsize=8)
    ax.set_ylabel("ms")
    ax.set_title(f"Render and two-view frame (G={sel[0]['grid_g']}, {sel[0]['resolution']} per view)\n{_label(rows)}",
                 fontsize=9)  # fmt: skip
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="*", help="Appendix C CSVs (default: newest bench/results/baseline_*.csv)")
    ap.add_argument("--out", default=str(HERE / "plots"))
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.csv] or sorted((HERE / "results").glob("baseline_*.csv"))[-1:]
    if not paths:
        raise SystemExit("no baseline CSV found; run `make bench` first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        rows = read_rows(path)
        tag = path.stem
        for p in (plot_scaling(rows, out / f"stage_scaling_{tag}.png"), plot_render_frame(rows, out / f"render_frame_{tag}.png")):
            if p:
                print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
