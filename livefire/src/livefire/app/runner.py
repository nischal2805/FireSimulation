"""Headless runner: capture threads -> simulation -> render -> HUD -> MP4 (PRD 5.1, 5.3, 5.5, 5.6).

Real-time mode: each display frame takes the newest frame per camera, calls ``sim.run_frame(wall_dt)`` (fixed
dt_sim, accumulator, never a variable dt), renders both views and writes the frame. Offline mode steps the sim
exactly ``round(1 / (fps * dt_sim))`` steps per output frame and labels the HUD "OFFLINE (not real-time)".
"""

from __future__ import annotations

import copy
import csv
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from livefire.app.hud import HudInfo, draw_hud, make_ladder
from livefire.app.lifecycle import Lifecycle, State
from livefire.app.replay import Event, apply_event, config_snapshot, write_record
from livefire.capture.synthetic_source import SyntheticCameraSource
from livefire.common.config import Params, load_params, load_yaml
from livefire.common.log import get_logger
from livefire.common.timers import StageTimer
from livefire.render.compositor import Compositor
from livefire.scene.builder import ViewInput, build_snapshot, manual_material
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

log = get_logger("livefire.app")
_P95_WINDOW = 120
_FPS_WINDOW = 30
_CSV_COLS = ["frame", "t_sim", "steps", "rt_factor", "sim_lag_s", "frame_ms", "sim_ms", "render_ms", "hud_ms",
             "write_ms", "latency_ms", "frame_age_ms", "degradation_level"]  # fmt: skip


@dataclass
class RunConfig:
    """Run options. ``seconds``: wall seconds (real-time) or simulated seconds (offline); ``frames`` overrides
    both when set. ``grid``: gas grid G override (None = grid.yaml)."""

    scene: str = "table_v1"
    seconds: float = 8.0
    frames: int | None = None
    out: Path = field(default_factory=lambda: Path("out/demo"))
    grid: int | None = None
    offline: bool = False
    fps: float | None = None
    events: list[Event] = field(default_factory=list)


def _pct(a: deque[float] | list[float], q: float) -> float:
    return float(np.percentile(np.asarray(a), q)) if len(a) else 0.0


class Runner:
    """Owns the lifecycle, camera sources, simulation and compositor. Single-threaded display loop."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.life = Lifecycle()
        self.app = load_yaml("app")
        self.fps = float(cfg.fps if cfg.fps is not None else self.app.get("fps", 30))
        self.grid_override: dict[str, Any] = {"G": cfg.grid} if cfg.grid else {}
        self.timer = StageTimer()
        self.level = 0
        self.ladder = make_ladder(float(self.app["degrade_volume_scale"]), float(self.app["degrade_step_factor"]))
        self.degradation_log: list[dict[str, Any]] = []
        self.sources: list[SyntheticCameraSource] = []
        self.sim: FireSimulation | None = None
        self.comp: Compositor | None = None
        self.summary: dict[str, Any] = {}

    # ---- lifecycle phases ---------------------------------------------------------------------
    def setup(self) -> None:
        """UNINITIALIZED -> CALIBRATING -> ACQUIRING -> READY."""
        try:
            self.life.transition(State.CALIBRATING)  # P0: synthetic calibration
            scene, cams = build_table_v1(), load_cameras()
            self.life.transition(State.ACQUIRING)  # P0: render the synthetic views, build the snapshot
            if self.cfg.scene != "table_v1":
                raise ValueError(f"unknown scene {self.cfg.scene!r}")
            views, rgb = {}, {}
            for cid, calib in cams.items():
                r = render_view(scene, calib, seed=0)
                views[cid] = ViewInput(calib, r.depth, r.valid, r.obj_id, r.rgb)
                rgb[cid] = r.rgb
            tab = scene.objects_table()
            self.snapshot = build_snapshot(
                views, {o: manual_material(m) for o, (_, m) in tab.items()},
                {**load_yaml("grid"), **self.grid_override}, names={o: nm for o, (nm, _) in tab.items()},
            )  # fmt: skip
            self.materials = MaterialTable.load()
            self.sim = FireSimulation(self.snapshot, self.materials, load_params("sim"), self.app)
            self.comp = Compositor(self.snapshot)
            self.sources = [SyntheticCameraSource(cid, rgb[cid], cams[cid].calib_version, self.fps) for cid in sorted(rgb)]
            self.life.transition(State.READY)
        except Exception:
            self.life.transition(State.ERROR)
            raise

    def _set_level(self, level: int, reason: str) -> None:
        lv = self.ladder[level]
        rp = load_params("render")
        e = copy.deepcopy({k: rp.meta(k) for k in rp.keys()})  # noqa: SIM118
        if lv.step_factor is not None:
            e["step_factor"]["value"] = lv.step_factor
        self.comp = Compositor(self.snapshot, Params(e), volume_scale=lv.volume_scale)
        self.level = level
        self.life.set_degraded(level > 0)
        rec = {"frame": self._frame, "t_sim": self.sim.t_sim, "level": level, "name": lv.name, "reason": reason,
               "volume_scale": self.comp.scale, "step_factor": float(self.comp.p["step_factor"])}  # fmt: skip
        self.degradation_log.append(rec)
        log.warning("DEGRADATION level %d (%s): %s", level, lv.name, reason)

    # ---- main loop ----------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Run the scenario; writes composite.mp4, timings.csv, timings_summary.json and the record files."""
        if self.life.state is State.UNINITIALIZED:
            self.setup()
        out = self.cfg.out
        out.mkdir(parents=True, exist_ok=True)
        sim, snap = self.sim, self.snapshot
        assert sim is not None
        offline = self.cfg.offline
        dt = sim.dt_sim
        steps_per_frame = max(1, round(1.0 / (self.fps * dt)))
        n_frames = self.cfg.frames if self.cfg.frames is not None else (
            max(1, round(self.cfg.seconds * self.fps)) if offline else None)  # fmt: skip
        default_r = float(load_params("sim")["ignite_radius"])
        pending = sorted(self.cfg.events, key=lambda e: e.t)
        applied: list[Event] = []
        cids = sorted(snap.cameras)
        w, h = snap.cameras[cids[0]].width, snap.cameras[cids[0]].height
        vw = cv2.VideoWriter(str(out / "composite.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w * len(cids), h))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open composite.mp4 (mp4v)")
        holes = {c: float(np.mean(snap.view_hole_mask[c])) for c in cids}
        for s in self.sources:
            s.start()
        self.life.transition(State.SIMULATING)
        frame_ms: deque[float] = deque(maxlen=_P95_WINDOW)
        stamps: deque[float] = deque(maxlen=_FPS_WINDOW + 1)
        rt_win: deque[tuple[float, float]] = deque(maxlen=60)
        low_rt = 0
        csv_f = open(out / "timings.csv", "w", newline="")  # noqa: SIM115
        wr = csv.writer(csv_f)
        wr.writerow(_CSV_COLS)
        t_loop0 = last = time.perf_counter()
        self._frame = 0
        rt = 0.0
        last_img = None
        try:
            while True:
                now = time.perf_counter()
                if n_frames is not None:
                    if self._frame >= n_frames:
                        break
                elif now - t_loop0 >= self.cfg.seconds and self._frame > 0:
                    break
                t0 = now
                wall_dt = (now - last) if self._frame > 0 else 1.0 / self.fps
                last = now
                frames = {s.cam_id: s.latest() for s in self.sources}
                if any(f is None for f in frames.values()):
                    raise RuntimeError("camera source produced no frame")
                age_ms = (time.perf_counter_ns() - min(f.t_arrive_ns for f in frames.values())) / 1e6
                # scripted events at sim time t (applied at a step boundary, recorded with its step index)
                while pending and pending[0].t <= sim.t_sim + 1e-9:
                    ev = apply_event(sim, pending.pop(0), default_r)
                    applied.append(ev)
                    log.info("event %s at step %d (t=%.3f s) n_lit=%s", ev.kind, ev.step, sim.t_sim, ev.n_lit)
                t_s = time.perf_counter()
                t_before = sim.t_sim
                if offline:
                    sim.step(steps_per_frame)
                    n_steps = steps_per_frame
                    rt_win.append((sim.t_sim - t_before, wall_dt))
                    ws = sum(b for _, b in rt_win)
                    rt, lag = (sum(a for a, _ in rt_win) / ws if ws > 0 else 0.0), 0.0
                else:
                    res = sim.run_frame(wall_dt)
                    n_steps, rt, lag = res.steps, res.rt_factor, res.sim_lag_s
                sim_ms = (time.perf_counter() - t_s) * 1e3
                self.timer.record("sim", sim_ms)
                # degradation ladder (real-time only): rt < threshold for N consecutive frames -> next level
                if not offline:
                    low_rt = low_rt + 1 if rt < float(self.app["degrade_rt_threshold"]) else 0
                    if low_rt >= int(self.app["degrade_frames"]) and self.level < len(self.ladder) - 1:
                        self._set_level(self.level + 1, f"rt_factor {rt:.3f} < {self.app['degrade_rt_threshold']} "
                                                         f"for {low_rt} frames")  # fmt: skip
                        low_rt = 0
                t_r = time.perf_counter()
                views = [self.comp.render(c, frames[c], sim) for c in cids]
                render_ms = (time.perf_counter() - t_r) * 1e3
                self.timer.record("render", render_ms)
                t_h = time.perf_counter()
                comp_img = np.concatenate(views, axis=1)
                p95 = _pct(frame_ms, 95)
                fps_deliv = (len(stamps) / (stamps[-1] - stamps[0])) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
                info = HudInfo(self.life.label, fps_deliv, p95, rt, sim.t_sim, self.ladder[self.level], holes,
                               len(snap.pos), offline)  # fmt: skip
                last_img = draw_hud(comp_img, info)
                hud_ms = (time.perf_counter() - t_h) * 1e3
                self.timer.record("hud", hud_ms)
                t_w = time.perf_counter()
                vw.write(np.ascontiguousarray(last_img[..., ::-1]))
                write_ms = (time.perf_counter() - t_w) * 1e3
                self.timer.record("write", write_ms)
                t_present = time.perf_counter_ns()
                latency_ms = (t_present - min(f.t_arrive_ns for f in frames.values())) / 1e6
                f_ms = (time.perf_counter() - t0) * 1e3
                frame_ms.append(f_ms)
                stamps.append(time.perf_counter())
                for k, v in (("frame", f_ms), ("latency", latency_ms), ("frame_age", age_ms)):
                    self.timer.record(k, v)
                wr.writerow([self._frame, f"{sim.t_sim:.5f}", n_steps, f"{rt:.4f}", f"{lag:.4f}", f"{f_ms:.2f}",
                             f"{sim_ms:.2f}", f"{render_ms:.2f}", f"{hud_ms:.2f}", f"{write_ms:.2f}",
                             f"{latency_ms:.2f}", f"{age_ms:.2f}", self.level])  # fmt: skip
                self._frame += 1
                if not offline:  # pace to the display fps; never sleeps in offline mode
                    spare = 1.0 / self.fps - (time.perf_counter() - t0)
                    if spare > 0:
                        time.sleep(spare)
        except Exception:
            self.life.transition(State.ERROR)
            raise
        finally:
            for s in self.sources:
                s.stop()
            vw.release()
            csv_f.close()
        wall = time.perf_counter() - t_loop0
        if last_img is not None:
            cv2.imwrite(str(out / "composite_last.png"), last_img[..., ::-1])
        digest = sim.state_host().digest()
        rest = list(pending)  # events never reached (t beyond the run) are reported, not silently dropped
        write_record(out, applied, config_snapshot(self.grid_override), {
            "seed": int(self.app["seed"]), "scene": self.cfg.scene, "final_step": sim.step_count,
            "t_sim": sim.t_sim, "digest": digest, "device": self.app["device"], "mode": "offline" if offline else "realtime",
            "grid_G": int(snap.solid_mask.shape[0]), "fps": self.fps, "frames": self._frame,
            "events_not_reached": [e.t for e in rest],
        })  # fmt: skip
        self.summary = {"frames": self._frame, "wall_s": wall, "t_sim": sim.t_sim, "rt_factor": rt, "digest": digest,
                        "final_degradation_level": self.level, "degradation_events": self.degradation_log,
                        "stages": self.timer.summary(), "gpu": "none", "framework": "warp-cpu"}  # fmt: skip
        (out / "timings_summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        return self.summary
