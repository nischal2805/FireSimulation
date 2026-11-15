"""M9 spread monotonicity (P2-003, PRD 11.1 M9, P2R9/P2R11).

Usage: python eval/p2_m9_spread.py [--split dev|heldout|scripted|both|all] [--quick] [--grid G] [--t-max S]
                                   [--out-dir DIR]

For every scenario (scripted demo scripts, randomized dev split, randomized held-out split) and every material in
``configs/scenarios.yaml: materials_compared`` set on the target object only (same geometry, same emitter, same gas
obstacles), runs the coupled simulation (``livefire.sim.scenarios.run_scenario``) and records the target's first
ignition time. Writes ``m9_spread_<date>.csv`` (one row per run, written incrementally) and ``m9_summary_<date>.csv``
(per ordered pair of ``ordered_pairs(MaterialTable)``: scenarios, violations, pass; unordered pairs are listed with
``asserted=false``; one ``verdict`` row). **M9 passes** iff there is no violation on the held-out split and at least
one target ignites in >= 75 % of the held-out scenarios (non-degenerate). A violation of a pair (A, B) is
``t_A > t_B + dt_sim`` (``None`` = never = +inf). Exit code 0 = pass (or a run without the held-out split, verdict
``not_evaluated``), 1 = fail. The ``notes`` column records t_max_s, G and the early-exit rule used. CPU numbers.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path
from typing import Any

from livefire.common.bench_csv import git_sha
from livefire.common.config import load_params, load_yaml
from livefire.sim.materials import MaterialTable
from livefire.sim.scenarios import (
    Scenario,
    monotonicity_violations,
    ordered_pairs,
    random_scenarios,
    run_scenario,
    scripted_scenarios,
    unordered_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
SPREAD_COLUMNS = [
    "scenario", "split", "seed", "gap_m", "angle_deg", "wind_m_s", "material", "t_ignite_s", "ignited", "wall_s",
    "end", "t_end_s", "n_lit", "max_T_g_K", "notes",
]  # fmt: skip
SUMMARY_COLUMNS = ["kind", "split", "material_a", "material_b", "asserted", "n_scenarios", "violations", "pass", "detail", "notes"]
NONDEGENERATE_MIN = 0.75


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def summarize(
    rows: list[dict[str, Any]], table: MaterialTable, dt: float, notes: str, cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Summary rows and the verdict (``PASS`` / ``FAIL`` / ``not_evaluated``) from the per-run ``rows``."""
    pairs = ordered_pairs(table, tuple(cfg["materials_compared"]))
    free = unordered_pairs(table, tuple(cfg["materials_compared"]))
    out: list[dict[str, Any]] = []
    splits = sorted({r["split"] for r in rows})
    times: dict[tuple[str, str], dict[str, float | None]] = {}
    for r in rows:
        times.setdefault((r["split"], r["scenario"]), {})[r["material"]] = r["t_ignite_s"] if r["ignited"] else None
    for sp in splits:
        sc = [t for (s, _), t in times.items() if s == sp]
        for a, b in pairs:
            n = sum(1 for t in sc if a in t and b in t)
            v = sum(1 for t in sc if monotonicity_violations(t, [(a, b)], dt))
            out.append({"kind": "pair", "split": sp, "material_a": a, "material_b": b, "asserted": "true",
                        "n_scenarios": n, "violations": v, "pass": str(v == 0).lower(), "detail": "", "notes": notes})  # fmt: skip
        for a, b in free:
            inf = float("inf")
            tt = [(t[a] if t[a] is not None else inf, t[b] if t[b] is not None else inf) for t in sc if a in t and b in t]
            a1 = sum(1 for x, y in tt if x < y - dt)
            b1 = sum(1 for x, y in tt if y < x - dt)
            out.append({"kind": "unordered", "split": sp, "material_a": a, "material_b": b, "asserted": "false",
                        "n_scenarios": len(tt), "violations": "", "pass": "",
                        "detail": f"{a} first in {a1}, {b} first in {b1}, tie/both-never in {len(tt) - a1 - b1}", "notes": notes})  # fmt: skip
    verdict = "not_evaluated"
    if "heldout" in splits:
        sc = [(s, t) for (s, _), t in times.items() if s == "heldout"]
        viol = sum(r["violations"] for r in out if r["kind"] == "pair" and r["split"] == "heldout")
        nd = sum(1 for _, t in sc if any(v is not None for v in t.values())) / max(1, len(sc))
        verdict = "PASS" if viol == 0 and nd >= NONDEGENERATE_MIN else "FAIL"
        out.append({"kind": "verdict", "split": "heldout", "material_a": "", "material_b": "", "asserted": "true",
                    "n_scenarios": len(sc), "violations": viol, "pass": verdict.lower() == "pass" and "true" or "false",
                    "detail": f"verdict {verdict}; held-out violations {viol}; fraction of scenarios where at least one "
                              f"target ignites {nd:.3f} (need >= {NONDEGENERATE_MIN})", "notes": notes})  # fmt: skip
    return out, verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all", choices=["dev", "heldout", "scripted", "both", "all"],
                    help="both = dev + heldout; all = scripted + dev + heldout (default)")  # fmt: skip
    ap.add_argument("--quick", action="store_true", help="1 dev scenario, 2 materials, t_max 5 s, G 16 (smoke run)")
    ap.add_argument("--grid", type=int, default=None, help="override the scenario grid G (default configs/scenarios.yaml)")
    ap.add_argument("--t-max", type=float, default=None, help="override t_max_s [s]")
    ap.add_argument("--out-dir", default=None, help="output folder (default eval/results)")
    args = ap.parse_args(argv)
    cfg = load_yaml("scenarios")
    if args.quick:
        cfg = {**cfg, "t_max_s": 5.0, "grid": {**cfg["grid"], "G": 16}, "materials_compared": cfg["materials_compared"][:2]}
        splits = ["dev"]
    else:
        splits = {"dev": ["dev"], "heldout": ["heldout"], "scripted": ["scripted"], "both": ["dev", "heldout"],
                  "all": ["scripted", "dev", "heldout"]}[args.split]  # fmt: skip
    if args.grid:
        cfg = {**cfg, "grid": {**cfg["grid"], "G": args.grid}}
    if args.t_max:
        cfg = {**cfg, "t_max_s": args.t_max}
    app = load_yaml("app")
    params, table = load_params("sim"), MaterialTable.load()
    dt = float(app["dt_sim"])
    today = datetime.date.today().isoformat()
    pre = "quick_" if args.quick else ""
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_spread, p_sum = out_dir / f"{pre}m9_spread_{today}.csv", out_dir / f"{pre}m9_summary_{today}.csv"
    notes = (f"t_max_s={cfg['t_max_s']}; G={cfg['grid']['G']}; dt_check_steps={cfg['dt_check_steps']}; rad_mode="
             f"{int(params['rad_mode'])}; ignition detected every step; run ends at the target's first ignition, when "
             f"nothing can ignite any more (extinct) or at t_max_s; CPU; git {git_sha()}"
             + ("; quick run, not an M9 result" if args.quick else ""))  # fmt: skip
    scen: list[Scenario] = []
    for sp in splits:
        scen += scripted_scenarios(cfg) if sp == "scripted" else random_scenarios(cfg, sp)
    if args.quick:
        scen = scen[:1]
    mats = list(cfg["materials_compared"])
    _log(f"M9: {len(scen)} scenarios x {len(mats)} materials, {notes}")
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with p_spread.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPREAD_COLUMNS)
        w.writeheader()
        for sc in scen:
            for m in mats:
                t1 = time.perf_counter()
                r = run_scenario(sc, m, params, app, cfg, table)
                wall = time.perf_counter() - t1
                row = {
                    "scenario": sc.name, "split": sc.split, "seed": sc.seed,
                    "gap_m": round(sc.meta["gap_m"], 5) if "gap_m" in sc.meta else "",
                    "angle_deg": round(sc.meta["angle_deg"], 3) if "angle_deg" in sc.meta else "",
                    "wind_m_s": round(sc.wind_speed, 4), "material": m,
                    "t_ignite_s": "" if r["t_ignite_s"] is None else round(r["t_ignite_s"], 5),
                    "ignited": str(r["ignited"]).lower(), "wall_s": round(wall, 1), "end": r["end"],
                    "t_end_s": round(r["t_end_s"], 4), "n_lit": r["n_lit"], "max_T_g_K": round(r["max_T_g"], 1),
                    "notes": notes,
                }  # fmt: skip
                w.writerow(row)
                f.flush()
                rows.append({**row, "t_ignite_s": r["t_ignite_s"], "ignited": r["ignited"]})
                _log(f"{sc.name:22s} {m:8s} t_ignite {r['t_ignite_s']} end {r['end']} ({r['t_end_s']:.2f} s) wall {wall:.1f} s")
    summ, verdict = summarize(rows, table, dt, notes, cfg)
    with p_sum.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        if s["kind"] in ("pair", "verdict"):
            _log(f"{s['kind']:8s} {s['split']:8s} {s['material_a']:8s} {s['material_b']:8s} n={s['n_scenarios']} violations={s['violations']} {s['detail']}")
    _log(f"M9 verdict: {verdict}; wall {(time.perf_counter() - t0) / 60:.1f} min; wrote {p_spread} and {p_sum}")
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
