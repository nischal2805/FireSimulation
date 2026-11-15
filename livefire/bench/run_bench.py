"""Benchmark harness (PRD 11.4, Appendix C): coupled sim, per-stage workloads, renders, two-view frame.

Usage: python bench/run_bench.py [--quick] [--device cpu|cuda:0] [--out CSV]
Device defaults to configs/app.yaml `device`. CPU rows are labelled gpu=none, framework=warp-<ver>-cpu.
Workloads are defined in bench/workloads.yaml (workload_version 2). Run with nothing else running.
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
import yaml

from livefire.capture.types import Frame
from livefire.common.bench_csv import BenchmarkRecorder, time_runner, vram_mb
from livefire.common.config import load_params, load_yaml
from livefire.render.compositor import Compositor
from livefire.scene.builder import snapshot_from_synthetic
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

HERE = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Bench:
    """Runs the workloads and appends one row per (stage, G / variant) to the recorder."""

    def __init__(self, cfg: dict[str, Any], device: str, rec: BenchmarkRecorder, quick: bool) -> None:
        self.cfg, self.device, self.rec, self.quick = cfg, device, rec, quick
        self.app = {**load_yaml("app"), "device": device}
        self.sync = lambda: wp.synchronize_device(device)
        self.warmup, self.repeats = int(cfg["warmup"]), int(cfg["repeats"])
        if self.repeats < 5 and not quick:
            raise SystemExit("PRD 11.4: at least 5 repeats")
        cores = f"CPU only, cores={os.cpu_count()}" if device == "cpu" else f"clocks: {rec.clocks}"
        self.calls: dict[str, int] = {}
        self.state_note = ""
        self.env_note = cores + ("; quick CI-size run, not a baseline" if quick else "")

    def _row(self, stage: str, variant: str, t: dict[str, float], spr: int, **fields: Any) -> dict[str, Any]:
        vram = vram_mb(self.device)
        notes = [
            f"first_launch_ms={t['first_launch_ms']:.1f}",
            f"repeats={self.repeats} samples_per_repeat={spr}",
            f"repeat_medians_ms=[{t['rep_median_min']:.3f},{t['rep_median_max']:.3f}]",
            *([fields.pop("notes")] if "notes" in fields else []),
            self.env_note,
            *([] if self.device == "cpu" else
              ["peak_vram_alloc_mb = Warp mempool high-water (process-cumulative; reserved not available)"]),
        ]
        row = self.rec.add_timing(
            stage, t, warmup=self.warmup, variant=variant, scene=self.cfg["scene"], mode="offline",
            peak_vram_alloc_mb=vram[0], peak_vram_reserved_mb=vram[1], notes="; ".join(notes), **fields,
        )  # fmt: skip
        _log(f"{stage:15s} {variant:34s} median {t['median_ms']:10.2f} ms  p95 {t['p95_ms']:10.2f} ms  n={int(t['n'])}")
        return row

    def _time(self, run: Any, spr: int) -> dict[str, float]:
        return time_runner(run, self.sync, self.warmup, self.repeats, spr)

    def make_sim(self, g: int) -> tuple[FireSimulation, Any]:
        snap = snapshot_from_synthetic(self.cfg["scene"], {"G": g})
        sim = FireSimulation(snap, MaterialTable.load(), load_params("sim"), self.app)
        ig = self.cfg["ignite"]
        n_lit = sim.ignite(np.asarray(ig["point"], dtype=float), float(ig["radius"]))
        if n_lit == 0:
            raise RuntimeError("ignition lit no elements; benchmark would measure a cold scene")
        t0 = time.perf_counter()
        sim.step(int(self.cfg["preroll_steps"]))
        _log(f"G={g}: {sim.surface.n} elements, {n_lit} lit, pre-roll {self.cfg['preroll_steps']} steps "
             f"in {time.perf_counter() - t0:.1f} s")  # fmt: skip
        return sim, snap

    def sim_stages(self, g: int, sim: FireSimulation) -> None:
        spr = int(self.cfg["sim_samples_per_repeat"][g])
        dt, surf, gas = sim.dt_sim, sim.surface, sim.gas
        n_iters = int(sim.params["n_iters"])
        stages = sim.stages(dt)
        pre = f"pre-roll {self.cfg['preroll_steps']} steps"
        back = "component rows run back-to-back on the evolving state (sim_step first, then the components)"
        self.calls = {"surface": 0, "gas": 0}

        def counted(name: str) -> Any:
            def run() -> None:
                self.calls[name] += 1
                stages[name]()

            return run

        def row(stage: str, variant: str, run: Any) -> None:
            t = self._time(run, spr)
            burning = int(surf.to_host()["burning"].sum())  # state at the end of this stage's timing
            self._row(stage, variant, t, spr, grid_g=g, surface_elements=surf.n,
                      notes=f"{pre}; {burning} elements burning after this stage; {back}")  # fmt: skip

        row("sim_step", "coupled_1step", lambda: sim.step(1))
        row("sample_gas_step", "sample_T_at_elements", stages["sample_gas"])
        row("surface_step", "thermal+combustion", counted("surface"))
        row("gas_step", "full_with_sources", counted("gas"))
        row("projection", f"rbgs_iters={n_iters}", lambda: gas.project(dt, n_iters))
        self.state_note = (
            f"state: G={g}, {pre}, sim_step_count={sim.step_count} (pre-roll + sim_step warmup/samples), then "
            f"{self.calls['surface']} surface-only and {self.calls['gas']} gas-only component calls "
            f"(they do not advance step_count), {int(surf.to_host()['burning'].sum())} elements burning"
        )  # fmt: skip

    def render_stages(self, g: int, sim: FireSimulation, snap: Any) -> None:
        scene, cams = build_table_v1(), load_cameras()
        cids = sorted(snap.cameras)
        frames = {
            c: Frame(c, render_view(scene, cams[c], seed=0).rgb, time.perf_counter_ns(), 0, cams[c].calib_version)
            for c in cids
        }  # fmt: skip
        cam = snap.cameras[cids[0]]
        res = f"{cam.width}x{cam.height}"
        spr = int(self.cfg["render_samples_per_repeat"])
        base = {"grid_g": g, "surface_elements": sim.surface.n, "resolution": res}
        st = f"renders the evolved {self.state_note}"
        for vs in self.cfg["render_volume_scales"]:
            comp = Compositor(snap, volume_scale=float(vs), device=self.device)
            t = self._time(lambda c=comp: c.render(cids[0], frames[cids[0]], sim), spr)
            self._row("render_view", f"volume_scale={vs}", t, spr, views=1, notes=st, **base)
        comp = Compositor(snap, device=self.device)
        vs = comp.scale
        spr = int(self.cfg["frame_two_views_samples_per_repeat"])
        wall_dt = float(self.cfg["frame_wall_dt"])
        steps: list[int] = []

        def frame() -> None:
            steps.append(sim.run_frame(wall_dt).steps)
            np.concatenate([comp.render(c, frames[c], sim) for c in cids], axis=1)

        t = self._time(frame, spr)
        n = int(np.median(steps))
        self._row(
            "frame_two_views", f"run_frame+2xrender+concat vs={vs}", t, spr, views=len(cids),
            delivered_fps=round(1000.0 / t["median_ms"], 4),
            rt_factor=round(n * sim.dt_sim / (t["median_ms"] / 1e3), 4),
            notes=f"{n} sim steps per frame (wall_dt={wall_dt:.4f} s); delivered_fps and rt_factor from the median; "
                  f"no HUD or video write; {st}; run_frame advances the sim during timing", **base,
        )  # fmt: skip

    def run(self) -> None:
        render_g = int(self.cfg["render_grid"])
        render_done = False
        for g in self.cfg["sim_grids"]:
            sim, snap = self.make_sim(int(g))
            self.sim_stages(int(g), sim)
            if int(g) == render_g:
                self.render_stages(int(g), sim, snap)
                render_done = True
            del sim
        if not render_done:  # render_grid not among sim_grids: run on its own simulation
            sim, snap = self.make_sim(render_g)
            self.render_stages(render_g, sim, snap)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="CI-size run (bench/workloads.yaml `quick`)")
    ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: configs/app.yaml device)")
    ap.add_argument("--out", default=None, help="CSV path (default bench/results/{baseline|quick}_<dev>_<date>.csv)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((HERE / "workloads.yaml").read_text())
    if args.quick:
        cfg = {**cfg, **cfg["quick"]}
    cfg.pop("quick", None)
    device = args.device or load_yaml("app").get("device", "cpu")
    today = datetime.date.today().isoformat()
    dev_tag = "cpu" if device == "cpu" else "gpu"
    kind = "quick" if args.quick else "baseline"
    out = Path(args.out) if args.out else HERE / "results" / f"{kind}_{dev_tag}_{today}.csv"
    rec = BenchmarkRecorder(device, run_id=f"{kind}-{dev_tag}-{today}", workload_version=cfg["workload_version"])
    _log(f"bench {kind} device={device} framework={rec.base['framework']} git_sha={rec.base['git_sha']} -> {out}")
    t0 = time.perf_counter()
    Bench(cfg, device, rec, args.quick).run()
    n = rec.flush(str(out))
    _log(f"wrote {n} rows to {out}; total wall {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
