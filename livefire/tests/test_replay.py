"""Runner smoke test, replay determinism and override behaviour (P0-010). Tiny grid, few frames."""

import json

import cv2
import numpy as np
import pytest

from livefire.app.replay import parse_ignite, parse_override, read_events, replay
from livefire.app.runner import RunConfig, Runner
from livefire.common.config import load_params, load_yaml
from livefire.scene.builder import snapshot_from_synthetic
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

PAPER = 3


def test_parse_events():
    e = parse_ignite("-0.075,0.0,0.1@0.3:0.06")
    assert e.point == (-0.075, 0.0, 0.1) and e.t == 0.3 and e.radius == 0.06
    assert parse_ignite("0,0,0@1").radius is None
    o = parse_override("3=metal@0.5")
    assert (o.obj_id, o.material, o.t) == (3, "metal", 0.5)
    with pytest.raises(ValueError):
        parse_override("3=unobtainium@0")
    with pytest.raises(ValueError):
        parse_ignite("garbage")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    ev = [parse_ignite("-0.075,0.0,0.1@0.0:0.06"), parse_override("2=wood@0.05")]
    s = Runner(RunConfig(out=out, grid=32, frames=10, offline=True, events=ev)).run()
    return out, s


def test_runner_smoke(run_dir):
    out, s = run_dir
    for f in ("composite.mp4", "timings.csv", "events.jsonl", "config_snapshot.yaml", "run.json", "composite_last.png"):
        assert (out / f).is_file(), f
    cap = cv2.VideoCapture(str(out / "composite.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (2560, 720)
    run = json.loads((out / "run.json").read_text())
    assert run["seed"] == load_yaml("app")["seed"] and run["final_step"] == 20 and run["digest"] == s["digest"]
    assert len(read_events(out / "events.jsonl")) == 2
    assert s["stages"]["frame"]["n"] == 10 and s["stages"]["latency"]["median_ms"] > 0


def test_replay_digest_matches(run_dir):
    out, _ = run_dir
    r = replay(out)
    assert r["ok"] and r["steps"] == 20


def test_replay_detects_mismatch(run_dir, tmp_path):
    out, _ = run_dir
    for f in out.iterdir():
        if f.is_file() and f.suffix != ".mp4":
            (tmp_path / f.name).write_bytes(f.read_bytes())
    ev = (tmp_path / "events.jsonl").read_text().splitlines()
    (tmp_path / "events.jsonl").write_text(ev[1] + "\n")  # drop the ignition
    assert not replay(tmp_path)["ok"]


def test_metal_override_prevents_burning():
    snap = snapshot_from_synthetic("table_v1", {"G": 16})
    table = MaterialTable.load()
    paper = snap.obj_id == PAPER
    top = snap.pos[np.flatnonzero(paper)[np.argmax(snap.pos[paper, 2])]]

    def go(override: bool):
        sim = FireSimulation(snap, table, load_params("sim"), load_yaml("app"))
        if override:
            sim.set_material(PAPER, table.material_id("metal"))
        n = sim.ignite(top, 0.06)
        sim.step(30)
        st = sim.state_host()
        return n, st.burning[paper].sum(), (st.char[paper] + st.removed[paper]).max()

    n0, burn0, dmg0 = go(False)
    n1, burn1, dmg1 = go(True)
    assert n0 > 0 and burn0 > 0 and dmg0 > 0
    assert n1 == 0 and burn1 == 0 and dmg1 == 0


def test_replay_rejects_out_of_order_events(run_dir, tmp_path):
    out, _ = run_dir
    for f in out.iterdir():
        if f.is_file() and f.suffix != ".mp4":
            (tmp_path / f.name).write_bytes(f.read_bytes())
    ev = (tmp_path / "events.jsonl").read_text().splitlines()
    (tmp_path / "events.jsonl").write_text("\n".join(reversed(ev)) + "\n")  # override (step 4) before ignite (0)
    with pytest.raises(ValueError, match="before the current step"):
        replay(tmp_path)


def test_unknown_scene_rejected_before_building():
    with pytest.raises(ValueError, match="unknown scene"):
        Runner(RunConfig(scene="nope"))


def test_failure_writes_record_and_stops_sources(tmp_path):
    r = Runner(RunConfig(out=tmp_path, grid=16, frames=5, offline=True))
    r.setup()

    def boom(*a, **k):
        raise KeyboardInterrupt

    r.comp.render = boom
    with pytest.raises(KeyboardInterrupt):
        r.run()
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["status"] == "interrupted" and (tmp_path / "events.jsonl").is_file()
    assert all(s._thread is None for s in r.sources)

    r2 = Runner(RunConfig(out=tmp_path / "e", grid=16, frames=5, offline=True))
    r2.setup()
    r2.comp.render = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    with pytest.raises(RuntimeError):
        r2.run()
    assert r2.life.state.value == "ERROR"
    assert json.loads((tmp_path / "e" / "run.json").read_text())["status"] == "error"


def test_replay_tolerance_branch(run_dir, tmp_path):
    """Non-CPU replay path (forced on CPU): compares final arrays from final_state.npz with a tolerance."""
    out, _ = run_dir
    assert (out / "final_state.npz").is_file()
    r = replay(out, force_tolerance=True)
    assert r["ok"] and r["mode"] == "tolerance"
    # a tiny (below tolerance) perturbation passes, a large one fails
    for f in out.iterdir():
        if f.is_file() and f.suffix != ".mp4":
            (tmp_path / f.name).write_bytes(f.read_bytes())
    z = dict(np.load(tmp_path / "final_state.npz"))
    z["T_g"] = z["T_g"] + 1e-4  # ~3e-7 relative at 293 K
    np.savez(tmp_path / "final_state.npz", **z)
    assert replay(tmp_path, force_tolerance=True)["ok"]
    z["T_g"] = z["T_g"] + 1.0
    np.savez(tmp_path / "final_state.npz", **z)
    assert not replay(tmp_path, force_tolerance=True)["ok"]
    cpu = load_yaml("app")["device"] == "cpu"
    # unforced: CPU digest ignores the perturbed npz; a GPU device compares the clean run with tolerance
    r = replay(tmp_path if cpu else out)
    assert r["ok"] and r["mode"] == ("digest" if cpu else "tolerance")
