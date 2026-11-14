"""bench/run_bench.py --quick end-to-end (slow) and bench/plots.py on a tiny synthetic CSV."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from livefire.common.bench_csv import REQUIRED, BenchmarkRecorder, read_rows
from livefire.common.config import load_yaml

BENCH = Path(__file__).resolve().parents[1] / "bench"


def _tiny_csv(path: Path, scale: float = 1.0) -> None:
    rec = BenchmarkRecorder("cpu", run_id="tiny", workload_version=1)
    for g in (32, 64):
        for stage in ("sim_step", "gas_step", "projection", "surface_step"):
            t = {"n": 5, "median_ms": g * scale, "p95_ms": g * scale * 1.1, "p99_ms": g * scale * 1.2}
            rec.add_timing(stage, t, warmup=1, scene="table_v1", mode="offline", grid_g=g)
    t = {"n": 5, "median_ms": 10.0, "p95_ms": 11.0, "p99_ms": 12.0}
    rec.add_timing("render_view", t, warmup=1, scene="table_v1", mode="offline", grid_g=32, variant="volume_scale=0.5",
                   resolution="1280x720", views=1)  # fmt: skip
    rec.flush(str(path))


def test_plots_all_csvs_one_series_each(tmp_path: Path) -> None:
    _tiny_csv(tmp_path / "baseline_a.csv")
    _tiny_csv(tmp_path / "baseline_b.csv", 2.0)
    r = subprocess.run(
        [sys.executable, str(BENCH / "plots.py"), str(tmp_path / "baseline_a.csv"), str(tmp_path / "baseline_b.csv"),
         "--out", str(tmp_path / "pl")],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    for name in ("stage_scaling.png", "render_frame.png"):
        f = tmp_path / "pl" / name
        assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", r.stdout
    import matplotlib

    matplotlib.use("Agg")
    sys.path.insert(0, str(BENCH))
    try:
        import plots

        runs = [(p.stem, read_rows(p)) for p in sorted(tmp_path.glob("baseline_*.csv"))]
        out = plots.plot_scaling(runs, tmp_path / "s.png")
        assert out.exists()
        import matplotlib.pyplot as plt

        assert plt.get_fignums() == []
    finally:
        sys.path.remove(str(BENCH))


@pytest.mark.slow
def test_run_bench_quick_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "quick.csv"
    device = str(load_yaml("app").get("device", "cpu"))  # run_bench defaults to the app.yaml device
    subprocess.run([sys.executable, str(BENCH / "run_bench.py"), "--quick", "--out", str(out)], check=True,
                   capture_output=True, text=True)  # fmt: skip
    rows = read_rows(out)
    assert [r["stage"] for r in rows] == [
        "sim_step", "sample_gas_step", "surface_step", "gas_step", "projection", "render_view", "frame_two_views",
    ]  # fmt: skip
    for r in rows:
        assert all(r[c] != "" for c in REQUIRED), r
        assert int(r["samples"]) >= 5
        if device == "cpu":
            assert r["gpu"] == "none" and r["framework"].endswith("-cpu")
        else:
            assert r["gpu"] not in ("", "none") and not r["framework"].endswith("-cpu")
    assert "elements burning" in rows[0]["notes"] and "evolved state" in rows[-1]["notes"]
