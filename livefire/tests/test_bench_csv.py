"""BenchmarkRecorder: Appendix C schema, row round-trip, required fields, CPU labelling."""

from __future__ import annotations

from pathlib import Path

import pytest

from livefire.common import interfaces
from livefire.common.bench_csv import COLUMNS, REQUIRED, BenchmarkRecorder, read_rows, time_runner


def _prd_columns() -> list[str]:
    prd = (Path(__file__).resolve().parents[1] / "docs" / "PRD.md").read_text()
    line = next(ln for ln in prd.splitlines() if ln.startswith("`run_id, date"))
    return [c.strip() for c in line.strip("`").split(",")]


def _recorder() -> BenchmarkRecorder:
    return BenchmarkRecorder("cpu", run_id="t-1", workload_version=1)


def test_columns_match_prd_appendix_c_in_order(tmp_path: Path) -> None:
    assert COLUMNS == _prd_columns()
    rec = _recorder()
    rec.record("sim_step", 12.5, "ms", scene="table_v1", mode="offline", grid_g=64)
    out = tmp_path / "b.csv"
    assert rec.flush(str(out)) == 1
    assert out.read_text().splitlines()[0].split(",") == COLUMNS


def test_implements_protocol() -> None:
    assert isinstance(_recorder(), interfaces.BenchmarkRecorder)


def test_row_round_trip_and_cpu_labels(tmp_path: Path) -> None:
    rec = _recorder()
    t = {"n": 15, "median_ms": 10.123456, "p95_ms": 12.5, "p99_ms": 13.0}
    rec.add_timing("gas_step", t, warmup=2, scene="table_v1", mode="offline", grid_g=96, variant="v",
                   notes="a, b; c", resolution="1280x720")  # fmt: skip
    rec.record("x", 3.0, "ms", scene="s", mode="m", custom_label="q")
    out = tmp_path / "b.csv"
    rec.flush(str(out))
    rows = read_rows(out)
    r = rows[0]
    assert (r["stage"], r["grid_g"], r["samples"], r["warmup_frames"]) == ("gas_step", "96", "15", "2")
    assert (r["median_ms"], r["p95_ms"], r["p99_ms"]) == ("10.1235", "12.5", "13.0")
    assert r["notes"] == "a, b; c" and r["resolution"] == "1280x720"  # CSV quoting round-trips
    assert (r["gpu"], r["driver"], r["cuda"], r["power_mode"]) == ("none", "none", "none", "unknown")
    assert r["framework"].startswith("warp-") and r["framework"].endswith("-cpu")
    assert r["peak_vram_alloc_mb"] == "" and r["peak_vram_reserved_mb"] == ""
    assert rows[1]["median_ms"] == "3.0" and "custom_label=q" in rows[1]["notes"]


def test_required_fields_non_empty(tmp_path: Path) -> None:
    rec = _recorder()
    rec.add_timing("sim_step", {"n": 5, "median_ms": 1.0, "p95_ms": 1.1, "p99_ms": 1.2}, warmup=1,
                   scene="table_v1", mode="offline")  # fmt: skip
    rec.flush(str(tmp_path / "b.csv"))
    for row in read_rows(tmp_path / "b.csv"):
        empty = [c for c in REQUIRED if row[c] == ""]
        assert not empty, empty
        assert row["degradation_level"] == "0" and row["git_sha"] and row["date"]


def test_errors(tmp_path: Path) -> None:
    rec = _recorder()
    with pytest.raises(ValueError):
        rec.record("s", 1.0, "furlongs")
    with pytest.raises(ValueError):
        rec.add_row(not_a_column=1)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        rec.flush(str(blocker / "sub" / "x.csv"))  # parent is a regular file
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        read_rows(bad)


def test_time_runner_counts_and_discards_warmup() -> None:
    calls = []
    t = time_runner(lambda: calls.append(1), lambda: None, warmup=3, repeats=5, samples=2)
    assert len(calls) == 3 + 5 * 2 and t["n"] == 10
    assert t["median_ms"] <= t["p95_ms"] <= t["p99_ms"]
