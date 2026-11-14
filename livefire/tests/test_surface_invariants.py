"""Surface fire model invariants (P0-007, PRD 8.6) on the table_v1 snapshot and tiny element graphs."""

import copy

import numpy as np
import pytest
import warp as wp

from livefire.common.config import Params, load_params
from livefire.common.geometry import AABB
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import snapshot_from_synthetic
from livefire.scene.snapshot import ObjectRegistry, SceneSnapshot
from livefire.sim.device import select_device
from livefire.sim.materials import MaterialTable
from livefire.sim.surface import SurfaceFireModel

DEV = select_device()
MID = {n: i for i, n in enumerate(MATERIAL_CLASSES)}
S = 0.015  # tiny-graph spacing, m


@pytest.fixture(scope="module")
def snap():
    return snapshot_from_synthetic("table_v1")


@pytest.fixture(scope="module")
def table():
    return MaterialTable.load()


def sim_params(**over: float) -> Params:
    """configs/sim.yaml with value overrides; ranges widened to include the override."""
    p = load_params("sim")
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return Params(e)


def grid_graph(nx: int, ny: int, material: str | list[str], diag: bool = False) -> SceneSnapshot:
    """nx*ny elements on a plane (spacing S, normal +Z), 4-neighbour (optionally 8) symmetric graph."""
    n = nx * ny
    xy = np.array([(i * S, j * S) for j in range(ny) for i in range(nx)])
    pos = np.c_[xy, np.zeros(n)].astype(np.float32)
    offs = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diag else [])
    nb = -np.ones((n, len(offs)), np.int32)
    for j in range(ny):
        for i in range(nx):
            for s, (di, dj) in enumerate(offs):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny:
                    nb[j * nx + i, s] = b * nx + a
    names = [material] * n if isinstance(material, str) else material
    mid = np.array([MID[m] for m in names], np.uint8)
    normal = np.tile(np.array([0, 0, 1], np.float32), (n, 1))
    return SceneSnapshot(
        version=1, calib_version=0, roi=AABB(np.zeros(3), np.ones(3)), h_gas=0.03, cameras={},
        pos=pos, normal=normal, area=np.full(n, S * S, np.float32), obj_id=np.zeros(n, np.int32),
        material_id=mid, material_conf=np.ones(n, np.float32), neighbors=nb,
        solid_mask=np.zeros((1, 1, 1), np.uint8), solid_fraction=np.zeros((1, 1, 1), np.float32),
        view_depth={}, view_hole_mask={}, objects=ObjectRegistry(), surfel_spacing=S,
    )  # fmt: skip


def gas(values: np.ndarray | float, n: int | None = None) -> wp.array:
    a = np.full(n, values, np.float32) if n is not None else np.asarray(values, np.float32)
    return wp.array(a, dtype=wp.float32, device=DEV)


def test_sample_points_and_static_arrays(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(), DEV)
    sp = m.sample_points()
    assert sp.shape == snap.pos.shape
    np.testing.assert_allclose(sp, snap.pos + snap.normal * snap.h_gas, atol=1e-6)
    assert m.G.shape == snap.neighbors.shape and m.Cap.shape == (len(snap.pos),)
    assert np.all(m.G.numpy()[snap.neighbors < 0] == 0)
    assert m.substeps_for(1e-4) == 1


def test_uniform_grid_conduction_matches_prd(table):
    """Interior element of a uniform grid: G_ij = k delta per neighbour (PRD 8.2: A * k delta / s^2 with A = s^2)."""
    m = SurfaceFireModel(grid_graph(5, 5, "paper"), table, sim_params(), DEV)
    k = table.k[MID["paper"]] * table.delta[MID["paper"]]
    np.testing.assert_allclose(m.G.numpy()[12], k, rtol=1e-6)  # positions are float32


def test_fuel_bounds_over_burn(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(), DEV)
    m.set_material_ids(np.full(m.n, MID["paper"]))
    m0, m_res = table.m0[MID["paper"]], table.m_res[MID["paper"]]
    tg = gas(293.15, m.n)
    # ignite at a paper element of the real scene
    i0 = int(np.flatnonzero(snap.material_id == MID["paper"])[0])
    assert m.ignite(snap.pos[i0]) > 0
    prev = m.m_fuel.numpy().copy()
    dt = 0.05
    for k in range(int(120 / dt)):
        m.step(dt, tg)
        if k % 20 == 0 or k == int(120 / dt) - 1:
            s = m.to_host()
            assert np.all(s["m_fuel"] >= m_res - 1e-12) and np.all(s["m_fuel"] <= m0 + 1e-12)
            assert np.all(s["m_fuel"] <= prev + 1e-15), "fuel must be non-increasing"
            for c in ("char", "removed"):
                assert np.all((s[c] >= 0) & (s[c] <= 1))
            prev = s["m_fuel"]
    s = m.to_host()
    assert (s["m_fuel"] < m0).any(), "nothing burned"
    assert s["char"].max() == pytest.approx(1.0)  # paper burns to completion in 120 s


def test_no_spontaneous_fire(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(), DEV)  # real mixed materials
    tg = gas(293.15, m.n)
    for _ in range(600):
        m.step(1.0, tg)  # 600 s
    s = m.to_host()
    assert not s["burning"].any()
    assert np.all(s["E"] == 0)
    assert np.abs(s["T_s"] - 293.15).max() < 1e-3
    assert np.all(s["m_fuel"] == m.m0.numpy())
    assert m.clamp_events() == 0


@pytest.mark.parametrize("mat", ["metal", "glass", "ceramic", "other_nonflammable"])
def test_non_flammables_never_ignite(table, mat):
    m = SurfaceFireModel(grid_graph(4, 4, mat), table, sim_params(), DEV)
    assert m.ignite(np.array([0.02, 0.02, 0.0]), radius=1.0) == 0
    tg = gas(1500.0, m.n)
    for _ in range(600):
        m.step(0.1, tg)  # 60 s at 1500 K
    s = m.to_host()
    assert not s["burning"].any()
    assert np.all(s["m_fuel"] == 0) and np.all(s["char"] == 0) and np.all(s["removed"] == 0)
    assert np.all(s["E"] == 0)
    assert np.all(m.heat_W.numpy() == 0) and np.all(m.soot_kg_s.numpy() == 0)


def test_non_flammables_in_real_scene(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(), DEV)
    nonf = ~np.asarray(table.flammable)[snap.material_id]
    assert nonf.any()
    tg = gas(1500.0, m.n)
    p0 = snap.pos[np.flatnonzero(nonf)[0]]
    m.ignite(p0, radius=0.2)
    for _ in range(120):
        m.step(0.5, tg)  # 60 s
    s = m.to_host()
    assert not s["burning"][nonf].any()
    assert np.all(s["m_fuel"][nonf] == 0) and np.all(s["char"][nonf] == 0) and np.all(s["removed"][nonf] == 0)


def _no_loss(table: MaterialTable) -> MaterialTable:
    """Shallow copy of the table with eps = 0 (the loader rejects 0 only for fuel fields, not eps arrays)."""
    t = copy.copy(table)
    t.eps = np.zeros_like(table.eps)
    return t


@pytest.mark.parametrize("mixed", [False, True])
def test_conduction_energy_conservation(table, mixed):
    nx = ny = 6
    mats = ["metal" if (i % 3 == 0) else "paper" for i in range(nx * ny)] if mixed else "cloth"
    m = SurfaceFireModel(grid_graph(nx, ny, mats), _no_loss(table), sim_params(h_c=0.0), DEV)
    rng = np.random.default_rng(1234)
    t0 = 300.0 + 100.0 * rng.random(m.n)  # < every T_ign: nothing burns
    m.T_s.assign(t0)
    cap = m.Cap.numpy()
    dt = 0.9 * m.dt_cond_max
    tg = gas(293.15, m.n)
    e0 = float(cap @ t0)

    def wvar(t):  # cap-weighted variance: the quantity diffusion dissipates
        tb = cap @ t / cap.sum()
        return float(cap @ (t - tb) ** 2)

    v_prev = wvar(t0)
    for _ in range(1000):
        m.step(dt, tg)
        assert m.last_substeps == 1
        t = m.T_s.numpy()
        v = wvar(t)
        assert v <= v_prev * (1 + 1e-12)
        v_prev = v
    t = m.T_s.numpy()
    err = abs(float(cap @ t) - e0) / e0
    print(f"\nconduction energy error (mixed={mixed}): {err:.3e}, variance {wvar(t0):.3e} -> {v_prev:.3e}")
    assert err < 1e-6
    assert v_prev < 0.5 * wvar(t0)
    assert not m.to_host()["burning"].any()
    if not mixed:  # uniform caps: plain variance also decreases
        assert t.var() < t0.var()


def test_stability_large_dt_substeps(table):
    # conduction: metal/paper mix, dt 100x the explicit limit
    m = SurfaceFireModel(grid_graph(6, 6, ["metal" if i % 2 else "paper" for i in range(36)]),
                         _no_loss(table), sim_params(h_c=0.0), DEV)  # fmt: skip
    rng = np.random.default_rng(0)
    t0 = 300.0 + 150.0 * rng.random(m.n)
    m.T_s.assign(t0)
    dt = 100.0 * m.dt_cond_max
    m.step(dt, gas(293.15, m.n))
    assert m.last_substeps >= 100 and m.last_substeps == m.substeps_for(dt)
    t = m.T_s.numpy()
    assert np.all(np.isfinite(t)) and t.min() >= t0.min() - 1e-9 and t.max() <= t0.max() + 1e-9  # max principle
    # convective/radiative: huge dt against a hot gas stays bounded between ambient and gas temperature
    m2 = SurfaceFireModel(grid_graph(3, 3, "paper"), table, sim_params(h_c=200.0), DEV)
    m2.step(30.0, gas(800.0, m2.n))
    assert m2.last_substeps > 1
    t2 = m2.T_s.numpy()
    assert np.all(np.isfinite(t2)) and t2.min() >= 293.15 and t2.max() <= 800.0 + 1e-6
    print(f"\nsub-steps: conduction dt/dt_max=100 -> {m.last_substeps}; convective dt=30 s h_c=200 -> {m2.last_substeps}")


def test_heat_and_soot_accounting(table):
    m = SurfaceFireModel(grid_graph(2, 2, "wood"), table, sim_params(), DEV)
    i = MID["wood"]
    m.ignite(np.zeros(3), radius=1.0)
    tg = gas(293.15, m.n)
    heat = soot = 0.0
    dt = 0.5
    for _ in range(2000):
        h, s = m.step(dt, tg)
        heat += float(h.numpy().sum()) * dt
        soot += float(s.numpy().sum()) * dt
    st = m.to_host()
    assert not st["burning"].any()
    used = table.m0[i] - table.m_res[i]
    area = m.n * S * S
    assert heat == pytest.approx(0.7 * table.chi[i] * table.dH[i] * used * area, rel=1e-4)
    assert soot == pytest.approx(table.y_soot[i] * used * area, rel=1e-4)
    assert np.all(st["m_fuel"] == table.m_res[i]) and np.all(st["char"] == 1.0)
    assert np.all(st["removed"] == 0)  # wood: not consumed_when_burnt


def test_ignition_order_paper_cloth_wood(table):
    """M9 seed (not the P2 validation): same geometry and gas-temperature ramp, ignition order by material."""
    times = {}
    for mat in ("paper", "cloth", "wood"):
        m = SurfaceFireModel(grid_graph(3, 3, mat), table, sim_params(), DEV)
        t, dt, t_ign = 0.0, 0.25, None
        while t < 600.0 and t_ign is None:
            tg = min(1000.0, 293.15 + 10.0 * t)  # 10 K/s ramp to 1000 K
            m.step(dt, gas(tg, m.n))
            t += dt
            if m.to_host()["burning"].any():
                t_ign = t
        times[mat] = t_ign
    print(f"\nignition times [s] (T_g ramp 10 K/s to 1000 K, h_c=25): {times}")
    assert all(v is not None for v in times.values())
    assert times["paper"] < times["cloth"] < times["wood"]


def _run_det(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(), DEV)
    m.set_material_ids(np.where(snap.material_id == MID["other_nonflammable"], MID["wood"], snap.material_id))
    m.ignite(snap.pos[0], radius=0.1)
    rng = np.random.default_rng(7)
    tg = gas(293.15 + 400.0 * rng.random(m.n), None)
    outs = []
    for _ in range(40):
        h, s = m.step(0.1, tg)
        outs.append((h.numpy().copy(), s.numpy().copy()))
    return m.to_host(), outs, m.clamp_events()


def test_determinism(snap, table):
    a, oa, ca = _run_det(snap, table)
    b, ob, cb = _run_det(snap, table)
    assert ca == cb
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k
    for (h1, s1), (h2, s2) in zip(oa, ob):
        assert h1.tobytes() == h2.tobytes() and s1.tobytes() == s2.tobytes()
    assert a["burning"].any() or a["m_fuel"].min() < table.m0.max()


def test_ignite_radius_and_fire_state(table):
    m = SurfaceFireModel(grid_graph(5, 5, "paper"), table, sim_params(), DEV)
    n = m.ignite(np.array([2 * S, 2 * S, 0.0]), radius=1.01 * S)  # centre + 4 neighbours
    assert n == 5
    st = m.to_host()
    assert (st["T_s"] == 900.0).sum() == 5 and (st["E"] == table.t_delay[MID["paper"]]).sum() == 5
    m.step(1 / 60, gas(293.15, m.n))
    assert m.to_host()["burning"].sum() == 5
    fs = m.fire_state_host(np.zeros((2, 2, 2), np.float32), np.zeros((2, 2, 2, 3), np.float32),
                           np.zeros((2, 2, 2), np.float32), np.zeros((2, 2, 2), np.float32))  # fmt: skip
    assert fs.T_s.dtype == np.float32 and fs.burning.sum() == 5 and fs.step == 1




def test_ignite_survives_next_step_large_dt(table):
    m = SurfaceFireModel(grid_graph(5, 5, "paper"), table, sim_params(), DEV)
    n = m.ignite(np.array([2 * S, 2 * S, 0.0]), radius=1.01 * S)
    assert n == 5 and m.to_host()["burning"].sum() == 5  # burning set immediately
    m.step(0.5, gas(293.15, m.n))
    assert m.to_host()["burning"].sum() == 5
    assert m.to_host()["m_fuel"].min() < table.m0[MID["paper"]]
