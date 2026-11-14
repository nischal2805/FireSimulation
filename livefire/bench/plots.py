"""Benchmark plots (PRD 11.4: plots read only from CSVs).

Usage: python bench/plots.py [CSV ...] [--out DIR]
Default input: all bench/results/baseline_*.csv (one series per file, labelled gpu/framework/date/git sha).
Writes PNGs to bench/plots/: stage_scaling.png (stage median ms, p95 whiskers, vs gas grid G, log-log) and
render_frame.png (render_view and frame_two_views). CPU runs read `gpu=none, warp-<ver>-cpu`.
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


Run = tuple[str, list[dict[str, str]]]  # (series label, rows of one CSV)


def _label(rows: list[dict[str, str]]) -> str:
    r = rows[0]
    return f"gpu={r['gpu']}, {r['framework']}, {r['date']}"


def plot_scaling(runs: list[Run], out: Path) -> Path:
    """Stage ms vs G (log-log) for the sim stages; one colour per stage, one line style per CSV."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = ["-", "--", ":", "-."]
    gs: set[int] = set()
    for k, (label, rows) in enumerate(runs):
        for i, stage in enumerate(SCALING_STAGES):
            pts = sorted((int(r["grid_g"]), float(r["median_ms"]), float(r["p95_ms"])) for r in rows if r["stage"] == stage)
            if not pts:
                continue
            g, med, p95 = zip(*pts, strict=True)
            gs.update(g)
            ax.errorbar(g, med, yerr=[[0] * len(g), [b - a for a, b in zip(med, p95, strict=True)]], marker="o",
                        capsize=3, color=f"C{i}", ls=styles[k % len(styles)], label=f"{stage} [{label}]")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted(gs), [str(g) for g in sorted(gs)])
    ax.minorticks_off()
    ax.set_xlabel("gas grid G (cells per edge)")
    ax.set_ylabel("ms per call (median, whisker to p95)")
    ax.set_title("Stage cost vs grid size", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_render_frame(runs: list[Run], out: Path) -> Path | None:
    """Median bars (p95 marker) for render_view variants and frame_two_views, one bar per CSV."""
    keys: list[tuple[str, str]] = []
    for _, rows in runs:
        for r in rows:
            k = (r["stage"], r["variant"].split(" ")[0])
            if r["stage"] in ("render_view", "frame_two_views") and k not in keys:
                keys.append(k)
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 5))
    w = 0.8 / len(runs)
    for j, (label, rows) in enumerate(runs):
        by = {(r["stage"], r["variant"].split(" ")[0]): r for r in rows}
        xs = [i + j * w for i, k in enumerate(keys) if k in by]
        ax.bar(xs, [float(by[k]["median_ms"]) for k in keys if k in by], w, color=f"C{j}", label=f"median [{label}]")
        ax.plot(xs, [float(by[k]["p95_ms"]) for k in keys if k in by], "k_", ms=12)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(keys))], [f"{a}\n{b}" for a, b in keys], fontsize=8)
    ax.set_ylabel("ms (black tick = p95)")
    ax.set_title("Render and two-view frame", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="*", help="Appendix C CSVs (default: all bench/results/baseline_*.csv)")
    ap.add_argument("--out", default=str(HERE / "plots"))
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.csv] or sorted((HERE / "results").glob("baseline_*.csv"))
    if not paths:
        raise SystemExit("no baseline CSV found; run `make bench` first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[Run] = []
    for path in paths:
        rows = read_rows(path)
        runs.append((f"{_label(rows)}, git {rows[0]['git_sha']}", rows))
    for p in (plot_scaling(runs, out / "stage_scaling.png"), plot_render_frame(runs, out / "render_frame.png")):
        if p:
            print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
