"""Radiative heat transfer (P2-001, PRD 8.2, P2R1-P2R6): view factors, energy budget, spread, determinism."""

import copy

import numpy as np
import pytest
import warp as wp

from livefire.common.config import Params, load_params, load_yaml
from livefire.common.geometry import AABB
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import snapshot_from_synthetic
from livefire.scene.snapshot import ObjectRegistry, SceneSnapshot
from livefire.sim.coupled import FireSimulation
from livefire.sim.device import select_device
from livefire.sim.materials import MaterialTable
from livefire.sim.radiation import SIGMA, RadiationModel
from livefire.sim.surface import SurfaceFireModel

DEV = select_device()
MID = {n: i for i, n in enumerate(MATERIAL_CLASSES)}
S = 0.015  # tiny-graph spacing, m
PAPER, METAL, CLOTH = 3, 2, 4  # obj_ids of table_v1


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


def dev_arr(a, dtype):
    return wp.array(np.asarray(a), dtype=dtype, device=DEV)


def pair_q(table, rec_pos, rec_n, rec_mat="paper", mode=2, area=1e-4, **over):
    """q_rad at one receiver from one forced-burning emitter at the origin (normal +Z, T_s = T_rad_flame)."""
    p = sim_params(rad_mode=mode, **over)
    pos = np.array([[0, 0, 0], rec_pos], float)
    nrm = np.array([[0, 0, 1], rec_n], float)
    eps = np.array([0.9, table.eps[MID[rec_mat]]])
    two = np.array([True, table.consumed[MID[rec_mat]]])
    m = RadiationModel(pos, nrm, np.full(2, area), eps, two, AABB(np.full(3, -1.0), np.full(3, 1.0)), p, DEV)
    out = wp.zeros(2, dtype=wp.float64, device=DEV)
    m.compute(dev_arr([1, 0], wp.uint8), dev_arr([p["T_rad_flame"]] * 2, wp.float64), out)
    return out.numpy()[1], m, p, eps[1]


def test_analytic_view_factor_mode2(table):
    area = 1e-4
    q, _, p, eps_i = pair_q(table, [0, 0, 0.1], [0, 0, -1], area=area)
    ref = eps_i * p["eps_f"] * SIGMA * p["T_rad_flame"] ** 4 * area / (np.pi * 0.1**2)
    assert q == pytest.approx(ref, rel=1e-6)
    # receiver rotated by 60 deg about x: cos = 0.5
    a = np.deg2rad(60)
    q60, *_ = pair_q(table, [0, 0, 0.1], [0, -np.sin(a), -np.cos(a)], area=area)
    assert q60 == pytest.approx(0.5 * ref, rel=1e-6)
    # r < r_min saturates at r_min (cosines use the true direction, here on axis)
    r_min = p["rad_r_min"]
    qa, *_ = pair_q(table, [0, 0, 0.5 * r_min], [0, 0, -1], area=area)
    qb, *_ = pair_q(table, [0, 0, r_min], [0, 0, -1], area=area)
    assert qa == pytest.approx(qb, rel=1e-6) and qa > 0
    # beyond rad_R: nothing
    q_far, *_ = pair_q(table, [0, 0, 0.31], [0, 0, -1], area=area)
    assert q_far == 0.0
    q_in, *_ = pair_q(table, [0, 0, 0.29], [0, 0, -1], area=area)
    assert q_in > 0


def test_one_sided_vs_two_sided_receiver(table):
    full, *_ = pair_q(table, [0, 0, 0.1], [0, 0, -1], rec_mat="paper")
    # receiver at z=+0.1 with normal +z faces away from the emitter at the origin
    wood_away, *_ = pair_q(table, [0, 0, 0.1], [0, 0, 1], rec_mat="wood")
    paper_away, *_ = pair_q(table, [0, 0, 0.1], [0, 0, 1], rec_mat="paper")
    wood_facing, *_ = pair_q(table, [0, 0, 0.1], [0, 0, -1], rec_mat="wood")
    assert wood_away == 0.0
    assert paper_away == pytest.approx(full, rel=1e-12)
    assert wood_facing > 0


def _sheets():
    n = 10
    xy = np.array([(0.4 + i * S, 0.4 + j * S) for j in range(n) for i in range(n)])
    pos = np.r_[np.c_[xy, np.full(100, 0.4)], np.c_[xy, np.full(100, 0.55)]]
    nrm = np.tile([0.0, 0.0, 1.0], (200, 1))
    return pos, nrm


def test_mode1_vs_mode2_and_mode0(table):
    pos, nrm = _sheets()
    burning = np.r_[np.ones(100), np.zeros(100)].astype(np.uint8)
    res = {}
    for mode in (0, 1, 2):
        m = RadiationModel(pos, nrm, np.full(200, S * S), np.full(200, 0.9), np.ones(200, bool),
                           AABB(np.zeros(3), np.ones(3)), sim_params(rad_mode=mode), DEV)  # fmt: skip
        out = wp.zeros(200, dtype=wp.float64, device=DEV)
        m.compute(dev_arr(burning, wp.uint8), dev_arr(np.full(200, 600.0), wp.float64), out)
        q = out.numpy()
        res[mode] = float((q[100:] * S * S).sum())
        assert m.last_absorbed_W == pytest.approx(float((q * S * S).sum()), rel=1e-12)
    assert res[0] == 0.0
    assert res[2] > 0
    assert res[1] == pytest.approx(res[2], rel=0.15)


def test_energy_budget_p2r3(snap, table):
    """Per element and over several steps: heat_W(mode 0) - heat_W(mode 2) == P_rx * w_b / sum(w) for emitters.
    The paper burning set is identical in both twins until the cloth ignites (> 3 s) and the paper burns out
    (> 6 s), so 20 steps of 0.05 s (1 s) compare like with like; T_g is held at ambient."""
    dt = 0.05
    tg = dev_arr(np.full(len(snap.pos), 293.15, np.float32), wp.float32)
    m0 = SurfaceFireModel(snap, table, sim_params(rad_mode=0), DEV)
    m2 = SurfaceFireModel(snap, table, sim_params(rad_mode=2), DEV)
    for m in (m0, m2):
        assert m.ignite(np.array([-0.075, 0.0, 0.1]), radius=0.15) > 0
    for _ in range(20):
        m0.step(dt, tg)
        m2.step(dt, tg)
        r = m2.radiation
        w, sum_w, p_rx = r.w_all.numpy(), float(r.sum_w.numpy()[0]), r.last_absorbed_W
        assert p_rx > 0 and sum_w > 0
        h0, h2 = m0.heat_W.numpy().astype(np.float64), m2.heat_W.numpy().astype(np.float64)
        assert np.array_equal(m0.burning.numpy() > 0, m2.burning.numpy() > 0)
        np.testing.assert_allclose(h0 - h2, p_rx * w / sum_w, rtol=0, atol=1e-6 * h0.max())
        assert h0.sum() * dt == pytest.approx((h2.sum() + p_rx) * dt, rel=1e-6)
        assert m2.radiation_stats()["rad_deficit_J"] == 0.0
    assert m2.clamp_events() == 0


def _model(mode, pos, nrm, eps=0.9, two=True, roi=1.0, **over):
    n = len(pos)
    return RadiationModel(pos, nrm, np.full(n, S * S), np.full(n, eps), np.full(n, two),
                          AABB(np.zeros(3), np.full(3, roi)), sim_params(rad_mode=mode, **over), DEV)  # fmt: skip


def _q(m, burning, t=600.0):
    n = len(burning)
    out = wp.zeros(n, dtype=wp.float64, device=DEV)
    m.compute(dev_arr(burning, wp.uint8), dev_arr(np.full(n, t), wp.float64), out)
    return out.numpy()


def test_mode1_burning_receivers_in_burning_cells(table):
    """Self-cell subtraction. A lone burning element receives nothing from itself; an L-corner of two burning
    perpendicular sheets (every cell holds emitters, all receivers burning) agrees with mode 2 in total absorbed
    power within 25 % (mode 1 is inaccurate within ~1 rad_cell)."""
    one = np.array([[0.5, 0.5, 0.5], [0.58, 0.5, 0.5]])
    nrm1 = np.array([[0.0, 0, 1], [0.0, 0, 1]])
    q = _q(_model(1, one, nrm1), np.array([1, 0], np.uint8))
    assert q[0] == 0.0 and q[1] == 0.0  # coplanar receiver: cos = 0; self excluded
    one[1] = [0.58, 0.5, 0.55]
    nrm1[1] = [1.0, 0, 0]
    q = _q(_model(1, one, nrm1), np.array([1, 0], np.uint8))
    assert q[0] == 0.0 and q[1] > 0.0
    n = 10
    g = np.array([(0.4 + i * S, 0.4 + j * S) for j in range(n) for i in range(n)])
    burning = np.ones(200, np.uint8)
    # (i) two parallel burning sheets 0.15 m apart: every receiver sits in a burning cell
    pos, nrm = _sheets()
    q1, q2 = _q(_model(1, pos, nrm), burning), _q(_model(2, pos, nrm), burning)
    assert q2.sum() > 0 and q1.sum() == pytest.approx(q2.sum(), rel=0.15)
    # (ii) perpendicular burning sheets (horizontal z = 0.4, vertical x = 0.7), gap > 1 cell
    horiz = np.c_[g, np.full(100, 0.4)]
    vert = np.c_[np.full(100, 0.7), g[:, 0], g[:, 1] - 0.1]
    pos = np.r_[horiz, vert]
    nrm = np.r_[np.tile([0.0, 0, 1], (100, 1)), np.tile([1.0, 0, 0], (100, 1))]
    q1, q2 = _q(_model(1, pos, nrm), burning), _q(_model(2, pos, nrm), burning)
    assert q2.sum() > 0 and q1.sum() == pytest.approx(q2.sum(), rel=0.25)
    # (iii) near field (L-corner, adjacent elements share cells): mode 1 is only qualitative within ~1 rad_cell
    vert = np.c_[np.full(100, 0.4), g[:, 0], g[:, 1]]
    pos = np.r_[horiz, vert]
    q1 = _q(_model(1, pos, nrm), burning)
    assert np.isfinite(q1).all() and q1.sum() > 0


def test_mode1_cell_cap_overflow(table):
    pos, nrm = _sheets()
    burning = np.r_[np.ones(100), np.zeros(100)].astype(np.uint8)
    m = _model(1, pos, nrm, rad_max_emitters=16, rad_cell=0.02)
    q = _q(m, burning)
    assert m.overflow_count > 0 and m.last_emitters > 16
    assert np.isfinite(q).all() and q.sum() > 0


def test_mode1_one_sided_and_cutoff_match_mode2(table):
    for mat in ("paper", "wood"):
        for n_rec in ([0, 0, -1], [0, 0, 1], [0, -0.6, -0.8]):
            q1, *_ = pair_q(table, [0, 0, 0.1], n_rec, rec_mat=mat, mode=1)
            q2, *_ = pair_q(table, [0, 0, 0.1], n_rec, rec_mat=mat, mode=2)
            assert q1 == pytest.approx(q2, rel=1e-9, abs=0.0)
    assert pair_q(table, [0, 0, 0.1], [0, 0, 1], rec_mat="wood", mode=1)[0] == 0.0
    assert pair_q(table, [0, 0, 0.1], [0, 0, 1], rec_mat="paper", mode=1)[0] > 0.0
    assert pair_q(table, [0, 0, 0.31], [0, 0, -1], mode=1)[0] == 0.0  # beyond rad_R (centroid distance)
    assert pair_q(table, [0, 0, 0.29], [0, 0, -1], mode=1)[0] > 0.0


def test_reset_clears_overflow(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(rad_mode=2, rad_max_emitters=16), DEV)
    m.ignite(np.array([-0.075, 0.0, 0.1]), radius=0.15)
    m.step(0.05, dev_arr(np.full(m.n, 293.15, np.float32), wp.float32))
    assert m.radiation_stats()["rad_overflow"] > 0
    m.reset()
    assert m.radiation_stats()["rad_overflow"] == 0


def _spread(snap, table, mode, t_end=30.0, dt=0.05):
    m = SurfaceFireModel(snap, table, sim_params(rad_mode=mode), DEV)
    # light only the paper: elements of obj PAPER one by one (a single r = 0.15 ball would also reach the cloth edge)
    for i in np.flatnonzero(snap.obj_id == PAPER):
        m.ignite(snap.pos[i], radius=1e-3)
    cloth = snap.obj_id == CLOTH
    assert m.to_host()["burning"][cloth].sum() == 0
    tg = dev_arr(np.full(m.n, 293.15, np.float32), wp.float32)
    t_first, n_peak = None, 0
    for k in range(round(t_end / dt)):
        m.step(dt, tg)
        nb = int(m.burning.numpy()[cloth].sum())
        n_peak = max(n_peak, nb)
        if nb and t_first is None:
            t_first = (k + 1) * dt
    return m, t_first, n_peak


@pytest.mark.parametrize("mode", [2, 1])
def test_spread_paper_to_cloth(snap, table, mode):
    m, t_cloth, n_peak = _spread(snap, table, mode)
    s = m.to_host()
    print(f"\nrad_mode={mode}: first cloth ignition t = {t_cloth} s, peak burning cloth elements = {n_peak}")
    assert t_cloth is not None and t_cloth < 30.0
    metal = snap.obj_id == METAL
    assert not s["burning"][metal].any() and np.all(s["m_fuel"][metal] == 0)


def test_no_spread_without_radiation(snap, table):
    _m, t_cloth, _ = _spread(snap, table, 0)
    print(f"\nrad_mode=0: first cloth ignition t = {t_cloth}")
    assert t_cloth is None


def test_determinism_surface(snap, table):
    tg = dev_arr(np.full(len(snap.pos), 293.15, np.float32), wp.float32)
    runs = []
    for _ in range(2):
        m = SurfaceFireModel(snap, table, sim_params(rad_mode=2), DEV)
        m.ignite(np.array([-0.075, 0.0, 0.1]), radius=0.15)
        for _ in range(40):
            m.step(0.05, tg)
        runs.append(m.to_host())
    for k in runs[0]:
        assert np.array_equal(runs[0][k], runs[1][k]), k


def test_determinism_coupled_digest(table):
    s32 = snapshot_from_synthetic("table_v1", {"G": 32})
    digests = []
    for _ in range(2):
        sim = FireSimulation(s32, table, sim_params(rad_mode=2), load_yaml("app"))
        sim.ignite(np.array([-0.075, 0.0, 0.1]), 0.15)
        sim.step(6)
        digests.append(sim.state_host().digest())
        assert sim.stats()["rad_emitters"] > 0
    assert digests[0] == digests[1]


def test_overflow_counted_not_silent(snap, table):
    m = SurfaceFireModel(snap, table, sim_params(rad_mode=2, rad_max_emitters=16), DEV)
    assert m.ignite(np.array([-0.075, 0.0, 0.1]), radius=0.15) > 16
    tg = dev_arr(np.full(m.n, 293.15, np.float32), wp.float32)
    m.step(0.05, tg)
    st = m.radiation_stats()
    assert st["rad_overflow"] > 0 and st["rad_emitters"] > 16
    assert np.isfinite(m.to_host()["T_s"]).all()
