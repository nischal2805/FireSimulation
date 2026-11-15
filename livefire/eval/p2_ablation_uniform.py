"""Uniform vs material-dependent combustion ablation and demo-spread evidence (P2-003, PRD 11.2, P2R12).

Usage: python eval/p2_ablation_uniform.py [--part ablation|demo|both] [--quick] [--out-dir DIR]

Part ``ablation`` (``ablation_uniform_<date>.csv``): the scenes ``table_v1`` (G=32, the ``make run`` ignition) and
``demo_chain`` (scripted scenario), each with (a) material-dependent combustion (the committed table, the scene's own
materials) and (b) uniform combustion (P2R12): every non-table object uses one flammable material whose parameters
are the arithmetic mean of paper, cloth, wood and plastic (``consumed_when_burnt: false``); the snapshot keeps the
real materials (geometry and obstacles identical) and the ids are swapped with ``SurfaceFireModel.set_material_ids``
before the ignition. Same ignition in both variants. Per object: first ignition time, burnt area (elements that were
ever burning) at 10/30/60 s, energy released by the object, whether an originally non-flammable object ignited.

Part ``demo`` (``demo_spread_<date>.csv``): the ``make run`` path (snapshot from the rendered synthetic views via
``livefire.app.build``, grid 48) with the demo ignition at t = 0.3 s for the ``(T_rad_flame, radius)`` combinations of
``configs/scenarios.yaml: demo_ignition.compare`` (before and after P2-003 and neighbours): per object
the elements lit, first ignition time, burnt area, hottest temperature and the radiative flux/energy absorbed
(wood investigation: required energy ``C'' (T_ign - T_amb)`` is recorded for flammable objects). CPU numbers.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from livefire.app.build import acquire_snapshot, build_sim, load_synthetic_calibration
from livefire.common.bench_csv import git_sha
from livefire.common.config import load_params, load_yaml
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import snapshot_from_primitives, snapshot_from_synthetic
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable
from livefire.sim.scenarios import (
    Scenario,
    scenario_grid_cfg,
    scripted_scenarios,
    sim_params_with,
    uniform_table,
)

ROOT = Path(__file__).resolve().parents[1]
CHECK = (10.0, 30.0, 60.0)  # s, burnt-area report times
ABLATION_COLUMNS = [
    "scene", "variant", "object", "orig_material", "sim_material", "n_elements", "area_total_m2", "first_ignition_s",
    "burnt_area_10s_m2", "burnt_area_30s_m2", "burnt_area_60s_m2", "released_J", "rad_deficit_J", "ever_burning_elements",
    "nonflammable_object_ignited", "t_end_s", "wall_s", "notes",
]  # fmt: skip
DEMO_COLUMNS = [
    "variant", "T_rad_flame_K", "ignite_radius_m", "ignite_time_s", "grid_g", "object", "material", "n_elements", "n_lit_at_ignition",
    "first_ignition_s", "seconds_after_ignition", "burnt_area_6s_m2", "burnt_area_10s_m2", "max_T_s_K",
    "max_q_rad_W_m2", "max_absorbed_J_m2", "required_J_m2_T_ign", "t_end_s", "wall_s", "notes",
]  # fmt: skip


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _object_masks(snap: Any) -> dict[str, np.ndarray]:
    return {o.name: snap.obj_id == o.obj_id for o in snap.objects}


def run_variant(
    snap: Any, table: MaterialTable, params: Any, app: dict[str, Any], ignite: tuple[np.ndarray, float], t_end: float,
    uniform: bool,
) -> dict[str, Any]:
    """One coupled run of ``snap``; ``uniform`` swaps every non-``other_nonflammable`` element to the uniform class.
    Returns per-object results keyed by object name plus ``t_end_s``."""
    use, uid = uniform_table() if uniform else (MaterialTable.load(), 0)
    sim = FireSimulation(snap, use, params, app)
    orig_ids = snap.material_id.astype(np.int64)
    if uniform:
        ids = orig_ids.copy()
        ids[orig_ids != MATERIAL_CLASSES.index("other_nonflammable")] = uid
        sim.surface.set_material_ids(ids)
    n_lit = sim.ignite(ignite[0], ignite[1])
    if n_lit == 0:
        raise RuntimeError("ignition lit nothing")
    masks = _object_masks(snap)
    area = np.asarray(snap.area, np.float64)
    ever = sim.surface.burning.numpy() > 0
    first = {n: (0.0 if (ever & m).any() else None) for n, m in masks.items()}
    released = {n: 0.0 for n in masks}
    at: dict[float, dict[str, float]] = {}
    n_steps = round(t_end / sim.dt_sim)
    for k in range(1, n_steps + 1):
        sim.step(1)
        b = sim.surface.burning.numpy() > 0
        ever |= b
        p = sim.surface.released_power_W.numpy().astype(np.float64)
        t = k * sim.dt_sim
        for n, m in masks.items():
            if first[n] is None and (b & m).any():
                first[n] = t
            released[n] += float(p[m].sum()) * sim.dt_sim
        for c in CHECK:
            if c not in at and t >= c - 1e-9 and c <= t_end + 1e-9:
                at[c] = {n: float(area[ever & m].sum()) for n, m in masks.items()}
    return {"masks": masks, "first": first, "released": released, "at": at, "ever": ever, "area": area,
            "ids": sim.surface.material_id.copy(), "t_end_s": n_steps * sim.dt_sim,
            "rad_deficit_J": sim.stats()["rad_deficit_J"]}  # fmt: skip


def ablation_rows(t_end: float, notes: str, cfg: dict[str, Any], only: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Rows for table_v1 and demo_chain, material-dependent and uniform."""
    params, app = load_params("sim"), load_yaml("app")
    base = MaterialTable.load()
    di = cfg["demo_ignition"]
    scenes: list[tuple[str, Any, tuple[np.ndarray, float]]] = []
    scenes.append(("table_v1", snapshot_from_synthetic("table_v1", {"G": 32}),
                   (np.asarray(di["point"], float), float(di["radius_m"]))))  # fmt: skip
    sc: Scenario = next(s for s in scripted_scenarios(cfg) if s.name == "demo_chain")
    scenes.append(("demo_chain", snapshot_from_primitives(sc.objects, scenario_grid_cfg(cfg), sc.seed, base),
                   (np.asarray(sc.ignite_point, float), sc.ignite_radius)))  # fmt: skip
    rows: list[dict[str, Any]] = []
    for scene, snap, ign in scenes:
        if only and scene not in only:
            continue
        for variant in ("material_dependent", "uniform"):
            t0 = time.perf_counter()
            r = run_variant(snap, base, params, app, ign, t_end, variant == "uniform")
            wall = time.perf_counter() - t0
            for name, m in r["masks"].items():
                orig = MATERIAL_CLASSES[int(snap.material_id[m][0])]
                sim_m = MATERIAL_CLASSES[int(r["ids"][m][0])] if variant == "material_dependent" else "uniform_mean"
                nonflam = not base.flammable[MATERIAL_CLASSES.index(orig)]
                ever_n = int((r["ever"] & m).sum())
                row = {
                    "scene": scene, "variant": variant, "object": name, "orig_material": orig, "sim_material": sim_m,
                    "n_elements": int(m.sum()), "area_total_m2": round(float(r["area"][m].sum()), 6),
                    "first_ignition_s": "" if r["first"][name] is None else round(r["first"][name], 4),
                    "released_J": round(r["released"][name], 3), "rad_deficit_J": round(r["rad_deficit_J"], 6), "ever_burning_elements": ever_n,
                    "nonflammable_object_ignited": str(nonflam and ever_n > 0).lower(),
                    "t_end_s": round(r["t_end_s"], 3), "wall_s": round(wall, 1), "notes": notes,
                }  # fmt: skip
                for c in CHECK:
                    row[f"burnt_area_{int(c)}s_m2"] = round(r["at"][c][name], 6) if c in r["at"] else ""
                rows.append(row)
            _log(f"{scene:11s} {variant:18s} first ignition { {n: (None if v is None else round(v, 2)) for n, v in r['first'].items()} } wall {wall:.1f} s")  # fmt: skip
    return rows


def demo_rows(
    t_end: float, notes: str, cfg: dict[str, Any], combos: list[dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    """The ``make run`` path (grid 48) for each ``{T_rad_flame, radius_m}`` combination; the ignition is applied at
    ``time_s`` of the spec."""
    app = load_yaml("app")
    table = MaterialTable.load()
    di = cfg["demo_ignition"]
    scene_obj, cams = load_synthetic_calibration("table_v1")
    grid_cfg = {**load_yaml("grid"), "G": int(di["grid_g"])}
    snap, _ = acquire_snapshot(scene_obj, cams, grid_cfg)
    masks = _object_masks(snap)
    area = np.asarray(snap.area, np.float64)
    t_amb = float(load_params("sim")["T_amb"])
    out: list[dict[str, Any]] = []
    for combo in combos or di["compare"]:
        r_ign, t_flame = float(combo["radius_m"]), float(combo["T_rad_flame"])
        params = sim_params_with(T_rad_flame=t_flame)
        t0 = time.perf_counter()
        sim = build_sim(snap, table, params, app)
        dt = sim.dt_sim
        step_ign = round(float(di["time_s"]) / dt)
        sim.step(step_ign)
        t_ign = sim.t_sim
        lit = sim.surface.burning.numpy() > 0
        n_lit = sim.ignite(np.asarray(di["point"], float), r_ign)
        b0 = sim.surface.burning.numpy() > 0
        lit_by = {n: int((b0 & m).sum()) for n, m in masks.items()}
        ever = b0.copy()
        first = {n: (t_ign if (b0 & m).any() else None) for n, m in masks.items()}
        maxT = {n: float(sim.surface.T_s.numpy()[m].max()) for n, m in masks.items()}
        maxq = {n: 0.0 for n in masks}
        absorbed = np.zeros(snap.pos.shape[0])
        at: dict[float, dict[str, float]] = {}
        for k in range(step_ign + 1, round(t_end / dt) + 1):
            sim.step(1)
            b = sim.surface.burning.numpy() > 0
            ever |= b
            q = sim.surface.q_rad.numpy()
            absorbed += q * dt
            ts = sim.surface.T_s.numpy()
            t = k * dt
            for n, m in masks.items():
                if first[n] is None and (b & m).any():
                    first[n] = t
                maxT[n] = max(maxT[n], float(ts[m].max()))
                maxq[n] = max(maxq[n], float(q[m].max()))
            for c in (6.0, 10.0):
                if c not in at and t >= c - 1e-9:
                    at[c] = {n: float(area[ever & m].sum()) for n, m in masks.items()}
        wall = time.perf_counter() - t0
        assert not lit.any() and n_lit == int(b0.sum())
        for n, m in masks.items():
            mid = int(snap.material_id[m][0])
            req = float(table.C_areal[mid] * (table.T_ign[mid] - t_amb)) if table.flammable[mid] else None
            out.append({
                "variant": f"T{t_flame:g}_r{r_ign:g}", "T_rad_flame_K": t_flame, "ignite_radius_m": r_ign, "ignite_time_s": round(t_ign, 4),
                "grid_g": int(di["grid_g"]), "object": n, "material": MATERIAL_CLASSES[mid], "n_elements": int(m.sum()),
                "n_lit_at_ignition": lit_by[n], "first_ignition_s": "" if first[n] is None else round(first[n], 4),
                "seconds_after_ignition": "" if first[n] is None else round(first[n] - t_ign, 4),
                "burnt_area_6s_m2": round(at[6.0][n], 6) if 6.0 in at else "",
                "burnt_area_10s_m2": round(at[10.0][n], 6) if 10.0 in at else "",
                "max_T_s_K": round(maxT[n], 2), "max_q_rad_W_m2": round(maxq[n], 1),
                "max_absorbed_J_m2": round(float(absorbed[m].max()), 1),
                "required_J_m2_T_ign": "" if req is None else round(req, 1), "t_end_s": round(t_end, 3),
                "wall_s": round(wall, 1), "notes": notes,
            })  # fmt: skip
        _log(f"demo T_rad_flame {t_flame:g} radius {r_ign:g}: lit {lit_by}, first ignition { {n: (None if v is None else round(v, 2)) for n, v in first.items()} } wall {wall:.1f} s")  # fmt: skip
    return out


def _write(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    _log(f"wrote {len(rows)} rows to {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="both", choices=["ablation", "demo", "both"])
    ap.add_argument("--quick", action="store_true", help="2 s runs, table_v1 only, one radius (smoke run)")
    ap.add_argument("--t-end", type=float, default=None, help="simulated seconds of the ablation runs (default 60) / demo runs (10)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml("scenarios")
    today = datetime.date.today().isoformat()
    pre = "quick_" if args.quick else ""
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "eval" / "results"
    t0 = time.perf_counter()
    if args.part in ("ablation", "both"):
        t_end = 2.0 if args.quick else float(args.t_end or 60.0)
        notes = (f"t_end_s={t_end}; G=32 (scene ROI of table_v1 = configs/grid.yaml ROI, demo_chain = scenarios ROI); "
                 f"rad_mode=2; uniform = mean of paper/cloth/wood/plastic, consumed_when_burnt false; CPU; git {git_sha()}"
                 + ("; quick run" if args.quick else ""))  # fmt: skip
        rows = ablation_rows(t_end, notes, cfg, ("table_v1",) if args.quick else None)
        _write(out_dir / f"{pre}ablation_uniform_{today}.csv", ABLATION_COLUMNS, rows)
    if args.part in ("demo", "both"):
        t_end = 2.0 if args.quick else float(args.t_end or 10.0)
        notes = (f"t_end_s={t_end}; grid {cfg['demo_ignition']['grid_g']} (make run path: rendered synthetic views via "
                 f"app.build); rad_mode=2; CPU; git {git_sha()}" + ("; quick run" if args.quick else ""))  # fmt: skip
        combos = cfg["demo_ignition"]["compare"][:1] if args.quick else None
        _write(out_dir / f"{pre}demo_spread_{today}.csv", DEMO_COLUMNS, demo_rows(t_end, notes, cfg, combos))
    _log(f"wall time {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
