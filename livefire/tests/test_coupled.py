"""Coupled FireSimulation (P0-008): protocol conformance, accumulator, coupling, replay, long run, R9 sensitivity."""

import copy
import time

import numpy as np
import pytest

from livefire.common.config import Params, load_params, load_yaml
from livefire.common.interfaces import FireStateView, SimulationBackend
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import snapshot_from_synthetic
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

G = 32
MID = {n: i for i, n in enumerate(MATERIAL_CLASSES)}
PAPER, CLOTH = 3, 4  # obj_ids of the synthetic table_v1 scene


@pytest.fixture(scope="module")
def snap():
    return snapshot_from_synthetic("table_v1", {"G": G})


@pytest.fixture(scope="module")
def table():
    return MaterialTable.load()


def make(snap, table, **over: float) -> FireSimulation:
    p = load_params("sim")
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return FireSimulation(snap, table, Params(e), load_yaml("app"))


def paper_top(snap) -> np.ndarray:
    idx = np.flatnonzero(snap.obj_id == PAPER)
    return snap.pos[idx[np.argmax(snap.pos[idx, 2])]]


def test_protocol_conformance(snap, table):
    sim = make(snap, table)
    assert isinstance(sim, SimulationBackend)
    assert isinstance(sim, FireStateView)
    assert sim.roi is snap.roi and sim.h_gas == snap.h_gas
    assert sim.T_g.shape == (G, G, G) and sim.soot.shape == (G, G, G)
    n = len(snap.pos)
    assert sim.char.shape == (n,) and sim.removed.shape == (n,)
    assert sim.latency_ms == 0.0 and sim.peak_vram_allocated_mb == 0.0


def test_state_host_shapes_and_cell_centred_vel(snap, table):
    sim = make(snap, table)
    st = sim.state_host()
    assert st.vel.shape == (G, G, G, 3) and st.T_g.shape == (G, G, G) and st.p.shape == (G, G, G)
    assert st.T_s.shape == (len(snap.pos),) and st.t_sim == 0.0 and st.step == 0
    assert len(st.digest()) == 64


def test_step_counts_and_fixed_dt(snap, table):
    sim = make(snap, table)
    sim.step(3)
    assert sim.step_count == 3 and sim.t_sim == pytest.approx(3 / 60)
    assert sim.latency_ms > 0
    with pytest.raises(ValueError):
        sim.step(-1)


def test_run_frame_accumulator(snap, table):
    sim = make(snap, table)
    dt = sim.dt_sim
    r = sim.run_frame(0.4 * dt)  # not enough wall time for a step
    assert r.steps == 0 and r.sim_lag_s == pytest.approx(0.4 * dt)
    r = sim.run_frame(0.7 * dt)  # accumulated 1.1 dt -> one step, 0.1 dt left
    assert r.steps == 1 and r.sim_lag_s == pytest.approx(0.1 * dt, abs=1e-9)
    r = sim.run_frame(10 * dt)  # capped at max_substeps_per_frame
    assert r.steps == 4 and r.sim_lag_s == pytest.approx(6.1 * dt, abs=1e-9)
    assert sim.step_count == 5
    assert sim.t_sim == pytest.approx(5 * dt, abs=1e-12)  # dt never varies
    assert r.rt_factor == pytest.approx(5 * dt / (11.1 * dt), rel=1e-9)
    assert sim.stats()["sim_lag_s"] == pytest.approx(6.1 * dt, abs=1e-9)


def test_calls_before_reset_raise(snap, table):
    sim = make(snap, table)
    sim._ready = False
    with pytest.raises(RuntimeError):
        sim.step(1)
    with pytest.raises(RuntimeError):
        sim.ignite(np.zeros(3), 0.1)


def test_ignite_validation(snap, table):
    sim = make(snap, table)
    with pytest.raises(ValueError):
        sim.ignite(np.zeros(2), 0.1)
    with pytest.raises(ValueError):
        sim.ignite(np.zeros(3), 0.0)
    assert sim.ignite(np.array([5.0, 5.0, 5.0]), 0.05) == 0


def test_set_material_keeps_other_state(snap, table):
    sim = make(snap, table)
    assert sim.ignite(paper_top(snap), 0.03) > 0
    sim.step(5)
    before = sim.surface.to_host()
    cloth = snap.obj_id == CLOTH
    sim.set_material(CLOTH, MID["wood"])
    after = sim.surface.to_host()
    assert np.all(sim.surface.material_id[cloth] == MID["wood"])
    assert np.all(sim.surface.material_id[snap.obj_id == PAPER] == snap.material_id[snap.obj_id == PAPER])
    np.testing.assert_array_equal(after["burning"][~cloth], before["burning"][~cloth])
    np.testing.assert_array_equal(after["m_fuel"][~cloth], before["m_fuel"][~cloth])
    np.testing.assert_array_equal(after["T_s"], before["T_s"])
    assert np.all(after["m_fuel"][cloth] == table.m0[MID["wood"]])
    with pytest.raises(ValueError):
        sim.set_material(99, 0)
    with pytest.raises(ValueError):
        sim.set_material(CLOTH, 42)


def test_burning_paper_heats_gas_and_makes_soot(snap, table):
    sim = make(snap, table)
    assert sim.ignite(paper_top(snap), 0.03) > 0
    sim.step(60)
    st = sim.state_host()
    assert st.T_g.max() > 293.15 + 3.0
    assert st.soot.max() > 0
    assert np.isfinite(st.vel).all() and np.abs(st.vel[..., 2]).max() > 0.01
    assert st.m_fuel.min() < table.m0[MID["paper"]]
    s = sim.stats()
    assert s["src_lost"] == 0 and s["div_after"] < s["div_before"]


def run_scenario(snap, table, steps: int) -> str:
    sim = make(snap, table)
    sim.ignite(paper_top(snap), 0.03)
    sim.step(steps // 2)
    sim.ignite(np.array([-0.1, -0.1, 0.0]), 0.05)  # second event (cloth)
    sim.step(steps - steps // 2)
    return sim.state_host().digest()


def test_replay_digest_equal_on_cpu(snap, table):
    a = run_scenario(snap, table, 90)
    b = run_scenario(snap, table, 90)
    assert a == b
    sim = make(snap, table)  # different events -> different digest (the digest is not vacuous)
    sim.step(90)
    assert sim.state_host().digest() != a


def test_inflow_wind_changes_flow(snap, table):
    sim = make(snap, table, wind_speed=1.0)
    sim.step(5)
    assert sim.state_host().vel[..., 0].mean() > 0.05


@pytest.mark.slow
def test_long_run_10_min_bounded(snap, table):
    p = load_params("sim")
    sim = FireSimulation(snap, table, p, load_yaml("app"))
    assert sim.ignite(paper_top(snap), 0.03) > 0
    n_total, chunk = 36000, 600
    t0 = time.perf_counter()
    vmax = p["cfl_max"] * snap.h_gas / sim.dt_sim
    m0 = table.m0[snap.material_id]
    m_res = table.m_res[snap.material_id]
    peak = {"T_g": 0.0, "soot": 0.0, "vel": 0.0, "T_s": 0.0}
    for k in range(n_total // chunk):
        sim.step(chunk)
        st = sim.state_host()
        for name in ("T_g", "soot", "vel", "T_s", "p", "m_fuel", "char", "removed"):
            assert np.isfinite(getattr(st, name)).all(), f"{name} not finite at step {sim.step_count}"
        assert st.T_g.max() <= p["T_g_max"] and st.soot.min() >= 0
        assert st.T_g.min() >= p["T_amb"] - 1e-3
        vm = np.abs(st.vel).max()
        assert vm <= 1.5 * vmax, f"|u| {vm} exceeds clamp bound {vmax}"
        # Task 7 invariants still hold
        assert np.all(st.m_fuel >= m_res - 1e-6) and np.all(st.m_fuel <= m0 + 1e-6)
        assert np.all((st.char >= 0) & (st.char <= 1)) and np.all((st.removed >= 0) & (st.removed <= 1))
        assert st.T_s.max() <= p["T_s_max"] + 1e-3
        peak["T_g"] = max(peak["T_g"], float(st.T_g.max()))
        peak["soot"] = max(peak["soot"], float(st.soot.max()))
        peak["vel"] = max(peak["vel"], float(vm))
        peak["T_s"] = max(peak["T_s"], float(st.T_s.max()))
    wall = time.perf_counter() - t0
    s = sim.stats()
    print(f"\nlong run G={G}: {n_total} steps, wall {wall:.0f} s, peaks {peak}, clamp bound {vmax:.2f}, stats {s}")
    assert sim.step_count == n_total
    assert s["T_g_clamp"] == 0 or peak["T_g"] <= p["T_g_max"]


def cloth_response(snap, table, seconds: float, chunk: int = 12) -> dict[str, float | None]:
    """Light the upper half of the paper; return the simulated time at which any cloth element first burns
    (``ignition``) and at which the hottest cloth element first exceeds T_amb + 10 K (``heat_10K``, a continuous
    proxy: in the P0 baseline the cloth does not reach ignition from the gas alone). None = not reached."""
    sim = make(snap, table)
    assert sim.ignite(np.array([-0.075, 0.0, 0.14]), 0.08) > 0
    cloth = snap.obj_id == CLOTH
    out: dict[str, float | None] = {"ignition": None, "heat_10K": None}
    for _ in range(int(seconds / sim.dt_sim) // chunk):
        sim.step(chunk)
        if out["heat_10K"] is None and sim.surface.T_s.numpy()[cloth].max() >= 293.15 + 10.0:
            out["heat_10K"] = sim.t_sim
        if sim.surface.burning.numpy()[cloth].any():
            out["ignition"] = sim.t_sim
            break
    return out


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="R9: resolution sensitivity is reported, not gating, in P0")
def test_resolution_sensitivity_cloth_response_g32_vs_g64(table):
    """R9: cloth response with the paper lit, G=32 vs G=64 (G=64 vs G=128 on the GPU machine)."""
    r = {g: cloth_response(snapshot_from_synthetic("table_v1", {"G": g}), table, seconds=8.0) for g in (32, 64)}
    print(f"\ncloth response (simulated s; None = not reached in 8 s): G=32 {r[32]}, G=64 {r[64]}")
    for key in ("ignition", "heat_10K"):
        assert r[32][key] is not None and r[64][key] is not None, f"{key} not reached: {r}"
        assert abs(r[32][key] - r[64][key]) <= 0.25 * r[64][key]
