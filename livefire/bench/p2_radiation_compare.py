"""Radiation technique comparison (P2-002, ADR-0006): rad_mode 0 / 1 / 2 on table_v1.

Usage: python bench/p2_radiation_compare.py [--quick] [--out-bench CSV] [--out-effect CSV] [--no-coupled]

Writes two CSVs (CPU numbers; GPU cost is a pending user run):
- ``bench/results/radiation_cpu_<date>.csv`` (Appendix C): stage ``radiation`` (``SurfaceFireModel.compute_radiation``)
  and ``surface_step`` (one full ``SurfaceFireModel.step``), variant ``mode<k>_burning<n>``, 3 fire sizes reached by
  igniting radius 0.03 / 0.08 / 0.15 m (n = elements actually burning at the start of the timing).
- ``eval/results/radiation_effect_<date>.csv``: first-ignition time per object, burnt area per object at 10/30/60 s,
  for the whole paper sheet lit (``paper_all``) and the ``make run`` ignition (``local_r0.06``), on the surface model
  alone with the gas held at ambient (``surface``) and coupled at G=32 (``coupled_g32``).
The surface model alone uses T_g = T_amb so that the spread metric isolates radiation. Burnt = ever burning.
The wall time of the run is printed and written to the ``wall_s`` column of the effect CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from livefire.common.bench_csv import BenchmarkRecorder, time_runner
from livefire.common.config import Params, load_params, load_yaml
from livefire.scene.builder import snapshot_from_synthetic
from livefire.sim.coupled import FireSimulation
from livefire.sim.device import select_device
from livefire.sim.materials import MaterialTable
from livefire.sim.surface import SurfaceFireModel

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_VERSION = 2  # bench/workloads.yaml (P2R7)
IGNITE_POINT = np.array([-0.075, 0.0, 0.1])  # same point as bench/workloads.yaml and `make run`
# m: radii of the brief (~100 / ~1000 / ~3000 burning elements were expected, but table_v1 has only 280 paper, 288 cloth
# and 294 wood elements in total (whole-scene totals, not near the ignition point), so the actual counts are lower and
# are recorded in the variant); the last radius lights every flammable element (the largest fire the scene can hold).
FIRE_RADII = (0.03, 0.08, 0.15, 10.0)
OBJECTS = {
    0: "table",
    1: "wood_block",
    2: "metal_box",
    3: "paper_sheet",
    4: "cloth_patch",
}  # table_v1 obj ids
PAPER = 3
CHECKPOINTS = (10.0, 30.0, 60.0)  # s, burnt-area report times
EFFECT_COLUMNS = [
    "run_id", "date", "git_sha", "device", "model", "scenario", "rad_mode", "grid_g", "dt_sim_s", "t_end_s",
    "obj_id", "object", "material", "n_elements", "area_total_m2", "first_ignition_s", "burnt_area_10s_m2",
    "burnt_area_30s_m2", "burnt_area_60s_m2", "ever_burning_elements", "max_T_s_K", "rad_deficit_J", "wall_s",
]  # fmt: skip


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def params_with(**over: float) -> Params:
    """configs/sim.yaml with value overrides (ranges widened to include them)."""
    p = load_params("sim")
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return Params(e)


def ignite_paper(model: SurfaceFireModel, snap: Any) -> int:
    """Light every paper element (one by one, radius 1 mm so nothing else is hit): the Task 1 spread scenario."""
    n = 0
    for i in np.flatnonzero(snap.obj_id == PAPER):
        n += model.ignite(snap.pos[i], 1e-3)
    return n


def _fresh_surface(
    snap: Any, table: MaterialTable, mode: int, device: str
) -> tuple[SurfaceFireModel, wp.array]:
    model = SurfaceFireModel(snap, table, params_with(rad_mode=mode), device)
    tg = wp.array(np.full(model.n, model._p["T_amb"], np.float32), dtype=wp.float32, device=device)
    return model, tg


def cost_rows(
    rec: BenchmarkRecorder, snap: Any, table: MaterialTable, device: str, dt: float, quick: bool
) -> None:
    """(a) cost: median/p95 of compute_radiation and of a full surface step per mode and fire size."""
    warmup, repeats = (1, 1) if quick else (2, 5)
    spr_rad, spr_step = (1, 1) if quick else (10, 6)
    sync = lambda: wp.synchronize_device(device)
    radii = FIRE_RADII[:1] if quick else FIRE_RADII
    for mode in (0, 1, 2):
        for r in radii:
            model, tg = _fresh_surface(snap, table, mode, device)
            model.ignite(IGNITE_POINT, r)
            n_burn = int(model.to_host()["burning"].sum())
            var = f"mode{mode}_burning{n_burn}"
            for stage, run, spr in (
                ("radiation", model.compute_radiation, spr_rad),
                ("surface_step", lambda m=model, g=tg: m.step(dt, g), spr_step),
            ):
                t = time_runner(run, sync, warmup, repeats, spr)
                n_end = int(model.to_host()["burning"].sum())
                extra = (
                    f"rad_emitters={model.radiation_stats()['rad_emitters']:.0f}"
                    if mode
                    else "no radiation kernels"
                )
                rec.add_timing(
                    stage, t, warmup=warmup, variant=var, scene="table_v1", mode="offline", grid_g=_g(snap),
                    surface_elements=model.n,
                    notes=f"ignite radius {r} m at {IGNITE_POINT.tolist()}; {n_burn} burning at start, {n_end} at end; "
                          f"repeats={repeats} samples_per_repeat={spr}; first_launch_ms={t['first_launch_ms']:.1f}; "
                          f"T_g = T_amb; rad_mode={mode}; {extra}; dt={dt:.5f}; CPU only, cores={os.cpu_count()}"
                          + ("; quick run, not a baseline" if quick else ""),
                )  # fmt: skip
                _log(
                    f"{stage:13s} {var:22s} median {t['median_ms']:10.3f} ms  p95 {t['p95_ms']:10.3f} ms  n={int(t['n'])}"
                )


def _g(snap: Any) -> int:
    return round(float(load_yaml("grid")["G"]))


class _Tracker:
    """Per-object first ignition time, ever-burning mask, burnt-area checkpoints, max T_s."""

    def __init__(self, snap: Any, t_end: float) -> None:
        self.snap = snap
        self.n = len(snap.pos)
        self.ever = np.zeros(self.n, bool)
        self.first = {o: None for o in OBJECTS}
        self.area_at: dict[float, dict[int, float]] = {}
        self.max_T = {o: 0.0 for o in OBJECTS}
        self.checkpoints = [c for c in CHECKPOINTS if c <= t_end + 1e-9]
        self.area = np.asarray(snap.area, np.float64)

    def update(self, t: float, burning: np.ndarray, T_s: np.ndarray, initial: bool = False) -> None:
        b = burning > 0
        self.ever |= b
        for o in OBJECTS:
            m = self.snap.obj_id == o
            if self.first[o] is None and (b & m).any():
                self.first[o] = t
            self.max_T[o] = max(self.max_T[o], float(T_s[m].max()))
        for c in self.checkpoints:
            if c not in self.area_at and t >= c - 1e-9:
                self.area_at[c] = {
                    o: float(self.area[self.ever & (self.snap.obj_id == o)].sum()) for o in OBJECTS
                }


def effect_rows(
    snap: Any,
    table: MaterialTable,
    device: str,
    dt: float,
    t_end: float,
    *,
    quick: bool,
    coupled: bool,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """(b) effect: spread outcome per mode and ignition scenario, surface-only and (optionally) coupled G=32."""
    rows: list[dict[str, Any]] = []
    scenarios = [("paper_all", None), ("local_r0.06", 0.06)]
    models = [("surface", snap)] + (
        [("coupled_g32", snapshot_from_synthetic("table_v1", {"G": 32}))] if coupled else []
    )
    for model_kind, sn in models:
        for scen, radius in scenarios:
            for mode in (0, 1, 2):
                t0 = time.perf_counter()
                if model_kind == "surface":
                    sim_s, tg = _fresh_surface(sn, table, mode, device)
                    step = lambda s=sim_s, g=tg: s.step(dt, g)
                    surf = sim_s
                    fire = sim_s
                else:
                    app = {**load_yaml("app"), "device": device}
                    sim = FireSimulation(sn, table, params_with(rad_mode=mode), app)
                    step = lambda s=sim: s.step(1)
                    surf = sim.surface
                    fire = sim
                n_lit = ignite_paper(surf, sn) if radius is None else fire.ignite(IGNITE_POINT, radius)
                if n_lit == 0:
                    raise RuntimeError("ignition lit nothing")
                tr = _Tracker(sn, t_end)
                h = surf.to_host()
                tr.update(0.0, h["burning"], h["T_s"], initial=True)
                n_steps = round(t_end / dt)
                for k in range(1, n_steps + 1):
                    step()
                    h = surf.to_host()
                    tr.update(k * dt, h["burning"], h["T_s"])
                wall = time.perf_counter() - t0
                deficit = surf.radiation_stats()["rad_deficit_J"]
                for o, name in OBJECTS.items():
                    m = sn.obj_id == o
                    mat = table.names[int(sn.material_id[m][0])]
                    row = {
                        **meta, "model": model_kind, "scenario": scen, "rad_mode": mode, "grid_g": 32 if model_kind != "surface" else _g(sn),
                        "dt_sim_s": round(dt, 6), "t_end_s": t_end, "obj_id": o, "object": name, "material": mat,
                        "n_elements": int(m.sum()), "area_total_m2": round(float(tr.area[m].sum()), 6),
                        "first_ignition_s": "" if tr.first[o] is None else round(tr.first[o], 4),
                        "ever_burning_elements": int((tr.ever & m).sum()), "max_T_s_K": round(tr.max_T[o], 2),
                        "rad_deficit_J": round(deficit, 6), "wall_s": round(wall, 1),
                    }  # fmt: skip
                    for c in CHECKPOINTS:
                        row[f"burnt_area_{int(c)}s_m2"] = (
                            round(tr.area_at[c][o], 6) if c in tr.area_at else ""
                        )
                    rows.append(row)
                _log(f"effect {model_kind:11s} {scen:12s} mode {mode}: first ignition "
                     f"{ {OBJECTS[o]: (None if v is None else round(v, 2)) for o, v in tr.first.items()} }  wall {wall:.1f} s")  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 repeat, small fire, 2 s effect run (tests)")
    ap.add_argument("--no-coupled", action="store_true", help="skip the coupled G=32 effect variant")
    ap.add_argument("--out-bench", default=None)
    ap.add_argument("--out-effect", default=None)
    args = ap.parse_args(argv)
    device = select_device(None)
    today = datetime.date.today().isoformat()
    pre = "quick_" if args.quick else ""
    out_b = (
        Path(args.out_bench)
        if args.out_bench
        else ROOT / "bench" / "results" / f"{pre}radiation_cpu_{today}.csv"
    )
    out_e = (
        Path(args.out_effect)
        if args.out_effect
        else ROOT / "eval" / "results" / f"{pre}radiation_effect_{today}.csv"
    )
    dt = float(load_yaml("app")["dt_sim"])
    snap = snapshot_from_synthetic("table_v1")
    table = MaterialTable.load()
    rec = BenchmarkRecorder("cpu", run_id=f"radiation-cpu-{today}", workload_version=WORKLOAD_VERSION)
    meta = {"run_id": rec.base["run_id"], "date": today, "git_sha": rec.base["git_sha"], "device": "cpu"}
    t0 = time.perf_counter()
    _log(f"radiation compare: {snap.pos.shape[0]} elements, dt {dt:.5f}, quick={args.quick}")
    cost_rows(rec, snap, table, device, dt, args.quick)
    t_cost = time.perf_counter() - t0
    rows = effect_rows(
        snap, table, device, dt, 2.0 if args.quick else 60.0,
        quick=args.quick, coupled=not (args.quick or args.no_coupled), meta=meta,
    )  # fmt: skip
    n = rec.flush(str(out_b))
    out_e.parent.mkdir(parents=True, exist_ok=True)
    with out_e.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EFFECT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    _log(f"wrote {n} rows to {out_b}, {len(rows)} rows to {out_e}")
    _log(f"wall time: cost part {t_cost / 60:.1f} min, total {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
