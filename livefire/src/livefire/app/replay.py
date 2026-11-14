"""Scripted events, run recording and deterministic replay (PRD 5.5).

A run directory holds ``events.jsonl`` (one event per line, applied at a simulation step index),
``config_snapshot.yaml`` (every config the run used, incl. the grid override), ``run.json`` (seed, scene,
final step, final ``FireStateHost.digest()``) and ``final_state.npz`` (all final state arrays). Replay re-runs the
simulation only. CPU: exact digest match. Non-CPU device: float atomics may differ in the last bits, so the final
arrays are compared with ``np.allclose(rtol=GPU_RTOL, atol=GPU_ATOL)`` against ``final_state.npz``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from livefire.common.config import Params, config_dir, load_yaml, load_yaml_text
from livefire.common.log import get_logger
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

log = get_logger("livefire.app.replay")

GPU_RTOL, GPU_ATOL = 1e-5, 1e-6  # non-CPU replay tolerance on the final state arrays


def compare_states(expected_npz: Path, actual: Any, rtol: float = GPU_RTOL, atol: float = GPU_ATOL) -> tuple[bool, str]:
    """Compare ``actual`` (``FireStateHost``) against a saved ``final_state.npz``. Returns ``(ok, detail)``;
    ``detail`` names the worst array and its max abs difference."""
    ok, worst = True, ("", 0.0)
    with np.load(expected_npz) as z:
        if int(z["step"]) != int(actual.step):
            return False, f"step {int(actual.step)} != {int(z['step'])}"
        for name in actual._ARRAYS:
            a, b = np.asarray(z[name], np.float64), np.asarray(getattr(actual, name), np.float64)
            if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
                ok = False
            d = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float("inf")
            if d > worst[1]:
                worst = (name, d)
    return ok, f"max abs diff {worst[1]:.3g} in {worst[0] or '-'} (rtol={rtol}, atol={atol})"


@dataclass(frozen=True)
class Event:
    """A scripted event. ``t`` is the requested sim time [s]; ``step`` the sim step index at which it was
    applied (set when recorded; -1 before). ignite: ``point`` world (x,y,z) m, ``radius`` m.
    override: ``obj_id`` and ``material`` name. Interactive clicks (P1R5) are ignite events that also carry
    ``cam``, ``u``, ``v`` (the click pixel); replay only uses ``point`` and ``radius``."""

    kind: str  # "ignite" | "override"
    t: float
    step: int = -1
    point: tuple[float, float, float] | None = None
    radius: float | None = None
    obj_id: int | None = None
    material: str | None = None
    n_lit: int | None = None  # result, informational
    cam: int | None = None  # interactive click: camera id, full-resolution pixel (u, v) (informational; point rules)
    u: float | None = None
    v: float | None = None


def parse_ignite(spec: str) -> Event:
    """Parse ``"x,y,z@t[:r]"`` (metres, sim seconds, optional radius in metres)."""
    try:
        pos, rest = spec.rsplit("@", 1)
        tpart, _, rpart = rest.partition(":")
        x, y, z = (float(v) for v in pos.split(","))
        return Event("ignite", float(tpart), point=(x, y, z), radius=float(rpart) if rpart else None)
    except ValueError as e:
        raise ValueError(f"bad --ignite {spec!r}, expected 'x,y,z@t[:r]'") from e


def parse_override(spec: str) -> Event:
    """Parse ``"obj_id=material@t"``; material must be one of MATERIAL_CLASSES."""
    try:
        lhs, tpart = spec.rsplit("@", 1)
        oid, mat = lhs.split("=", 1)
        ev = Event("override", float(tpart), obj_id=int(oid), material=mat)
    except ValueError as e:
        raise ValueError(f"bad --override {spec!r}, expected 'obj_id=material@t'") from e
    if mat not in MATERIAL_CLASSES:
        raise ValueError(f"unknown material {mat!r}; one of {list(MATERIAL_CLASSES)}")
    return ev


def apply_event(sim: FireSimulation, ev: Event, default_radius: float) -> Event:
    """Apply ``ev`` to ``sim`` now; returns the recorded copy (step = current step count, n_lit filled)."""
    if ev.kind == "ignite":
        assert ev.point is not None
        r = ev.radius if ev.radius is not None else default_radius
        n = sim.ignite(np.asarray(ev.point, float), r)
        return replace(ev, step=sim.step_count, radius=r, n_lit=n)
    if ev.kind == "override":
        assert ev.obj_id is not None and ev.material is not None
        sim.set_material(ev.obj_id, MATERIAL_CLASSES.index(ev.material))
        return replace(ev, step=sim.step_count)
    raise ValueError(f"unknown event kind {ev.kind!r}")


def write_events(path: Path, events: list[Event]) -> None:
    path.write_text("".join(json.dumps(asdict(e)) + "\n" for e in events))


def read_events(path: Path) -> list[Event]:
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            if d.get("point") is not None:
                d["point"] = tuple(d["point"])
            out.append(Event(**d))
    return out


def config_snapshot(grid_override: dict[str, Any] | None) -> dict[str, Any]:
    """All configs a run depends on (app, sim, grid with override applied, render, materials)."""
    return {
        "app": load_yaml("app"),
        "sim": load_yaml("sim"),
        "grid": {**load_yaml("grid"), **(grid_override or {})},
        "render": load_yaml("render"),
        "materials": load_yaml_text((config_dir() / "materials.yaml").read_text()),
    }


def write_record(out: Path, events: list[Event], cfg: dict[str, Any], run: dict[str, Any]) -> None:
    """Write events.jsonl, config_snapshot.yaml and run.json (seed, scene, final step, digest, ...)."""
    out.mkdir(parents=True, exist_ok=True)
    write_events(out / "events.jsonl", events)
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def build_sim_from_snapshot(cfg: dict[str, Any], scene: str) -> tuple[Any, FireSimulation]:
    """Rebuild snapshot + FireSimulation from a recorded config snapshot with the same helpers Runner.setup
    uses (``livefire.app.build``). Returns ``(snapshot, sim)``."""
    from livefire.app.build import acquire_snapshot, build_sim, load_synthetic_calibration

    scene_obj, cams = load_synthetic_calibration(scene)
    snap, _ = acquire_snapshot(scene_obj, cams, cfg["grid"])
    return snap, build_sim(snap, MaterialTable(cfg["materials"]), Params(cfg["sim"]), cfg["app"])


def replay(run_dir: Path, render: bool = False, force_tolerance: bool = False) -> dict[str, Any]:
    """Re-run the recorded simulation (no rendering unless ``render``) and compare the final state.

    CPU (or ``force_tolerance=False`` on CPU): exact digest. Non-CPU device, or ``force_tolerance=True``: tolerance
    comparison against ``final_state.npz``. Returns ``{"ok": bool, "expected": str, "actual": str, "steps": int,
    "mode": "digest"|"tolerance", "detail": str}``. With ``render`` the final two-view
    composites are written to ``<run_dir>/replay_cam<i>.png``."""
    run = json.loads((run_dir / "run.json").read_text())
    cfg = yaml.load((run_dir / "config_snapshot.yaml").read_text(), Loader=yaml.SafeLoader)
    events = read_events(run_dir / "events.jsonl")
    if run.get("digest") is None:
        raise ValueError(f"{run_dir}: run has no digest (status {run.get('status')!r}); nothing to compare")
    tol = force_tolerance or str(cfg["app"].get("device", "cpu")) != "cpu"
    if tol:
        log.warning("replaying on device %s: comparing with tolerance rtol=%g atol=%g, not an exact digest",
                    cfg["app"].get("device"), GPU_RTOL, GPU_ATOL)
    snap, sim = build_sim_from_snapshot(cfg, run["scene"])
    default_r = float(cfg["sim"]["ignite_radius"]["value"])
    for ev in events:  # recorded order; steps must be non-decreasing
        if ev.step < sim.step_count:
            raise ValueError(f"event {ev.kind} at step {ev.step} is before the current step {sim.step_count} "
                             "(events.jsonl is out of order or corrupt)")
        sim.step(ev.step - sim.step_count)
        apply_event(sim, ev, default_r)
    sim.step(int(run["final_step"]) - sim.step_count)
    st = sim.state_host()
    actual = st.digest()
    if render:
        import cv2

        from livefire.capture.types import Frame
        from livefire.render.compositor import Compositor
        from livefire.scene.synthetic import build_table_v1, render_view

        comp, scene = Compositor(snap), build_table_v1()
        for cid, calib in snap.cameras.items():
            v = render_view(scene, calib, seed=0)
            img = comp.render(cid, Frame(cid, v.rgb, 0, 0, calib.calib_version), sim)
            cv2.imwrite(str(run_dir / f"replay_cam{cid}.png"), img[..., ::-1])
    if tol:
        ok, detail = compare_states(run_dir / "final_state.npz", st)
        mode = "tolerance"
    else:
        ok, detail, mode = actual == run["digest"], "exact digest", "digest"
    return {"ok": ok, "expected": run["digest"], "actual": actual, "steps": sim.step_count, "mode": mode,
            "detail": detail}
