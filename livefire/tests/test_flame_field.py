"""Combustion-intensity (flame) field (P0-008b, ADR-0004): gas scalar, surface power accessor, renderer use. CPU Warp, G<=32."""

import copy

import numpy as np
import pytest
import warp as wp

from livefire.common.config import Params, load_params, load_yaml
from livefire.common.geometry import project
from livefire.render.compositor import Compositor
from livefire.scene.builder import snapshot_from_synthetic
from livefire.sim.coupled import FireSimulation
from livefire.sim.device import select_device
from livefire.sim.gas import GasSolver
from livefire.sim.materials import MaterialTable

G = 32
PAPER = 3  # obj_id of the synthetic table_v1 paper
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DT = 1.0 / 60.0


def _override(name: str, **over: float) -> Params:
    p = load_params(name)
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return Params(e)


@pytest.fixture(scope="module")
def snap():
    return snapshot_from_synthetic("table_v1", {"G": G})


@pytest.fixture(scope="module")
def table():
    return MaterialTable.load()


def make(snap, table) -> FireSimulation:
    return FireSimulation(snap, table, load_params("sim"), load_yaml("app"))


def paper_top(snap) -> np.ndarray:
    idx = np.flatnonzero(snap.obj_id == PAPER)
    return snap.pos[idx[np.argmax(snap.pos[idx, 2])]]


def test_no_fire_flame_identically_zero(snap, table):
    sim = make(snap, table)
    for _ in range(4):
        sim.step(50)
        assert not sim.state_host().flame.any()
    assert sim.step_count == 200


def test_lit_patch_gives_flame_above_it(snap, table):
    sim = make(snap, table)
    pt = paper_top(snap)
    assert sim.ignite(pt, 0.03) > 0
    sim.step(30)  # 0.5 s
    f = sim.state_host().flame
    assert np.isfinite(f).all() and f.min() >= 0.0 and f.max() > 0.0
    assert not f[snap.solid_mask.astype(bool)].any()
    ci = ((pt - snap.roi.min) / snap.h_gas).astype(int)
    above = f[max(ci[0] - 2, 0) : ci[0] + 3, max(ci[1] - 2, 0) : ci[1] + 3, ci[2] : ci[2] + 4]
    assert above.max() > 0.5 * f.max()  # the peak sits just above the patch


def test_steady_source_converges_to_power_density():
    dev = select_device()
    h = 3.0 / G
    roi_min = np.array([-1.5, -1.5, -0.1])
    gas = GasSolver(G, h, roi_min, np.zeros((G,) * 3, np.uint8), load_params("sim"), dev, seed=1)
    cell = (16, 16, 16)
    pos = wp.array((roi_min + (np.array(cell) + 0.5) * h)[None, :].astype(np.float32), dtype=wp.float32, device=dev)
    zero = wp.zeros(1, dtype=wp.float32, device=dev)
    P = 2000.0  # W
    power = wp.array(np.array([P], np.float32), dtype=wp.float32, device=dev)
    for _ in range(120):  # 2 s = 12 flame lifetimes
        gas.step(DT, pos, zero, zero, power)
    f = gas.flame.numpy()
    target = P / h**3
    assert f[cell] == pytest.approx(target, rel=0.02)
    assert abs(f[cell] - target) / target < 0.02
    f2 = f.copy()
    f2[cell] = 0.0
    assert f2.max() < 1e-3 * target  # nothing leaks to neighbours (centre-of-cell source, zero velocity)


def test_flame_decays_after_burnout(snap, table):
    sim = make(snap, table)
    assert sim.ignite(paper_top(snap), 0.03) > 0
    sim.step(30)
    peak = sim.state_host().flame.max()
    assert peak > 0
    sim.surface.m_fuel.assign(sim.surface.m_res.numpy().copy())  # force burnout
    sim.step(120)  # 2 s
    assert sim.state_host().flame.max() < 1e-3 * peak


def _luma_near(L, uv, r=20):
    lum = L @ LUMA
    v, u = np.mgrid[0 : lum.shape[0], 0 : lum.shape[1]]
    return float(lum[(u - uv[0]) ** 2 + (v - uv[1]) ** 2 <= r**2].sum())


def test_flame_drives_visible_emission(snap, table, record_property):
    sim = make(snap, table)
    pt = paper_top(snap)
    # the whole paper sheet (radius 0.2 m): at G=32 a small patch gives flame < 0.1 flame_ref (see ADR-0004)
    assert sim.ignite(pt, 0.2) > 0
    sim.step(60)  # 1 s after ignition
    assert sim.state_host().T_g.max() < load_params("render")["T_emit_min"]  # T_g alone is below emission
    # Soot absorption/in-scatter switched off in both so that only emission is measured. flame_ref is lowered to
    # 1e5 W/m^3 for this G=32 test: with the default 1e6 the coarse-grid flame (<= 5e5 W/m^3) stays below the 700 K
    # emission threshold (total luminance ~6e-4, reported in ADR-0004).
    base = {"k_abs": 0.0, "soot_albedo": 0.0, "flame_ref": 1.0e5}
    on = Compositor(snap, params=_override("render", **base), volume_scale=0.5)
    off = Compositor(snap, params=_override("render", dT_flame=0.0, **base), volume_scale=0.5)
    for cid, calib in snap.cameras.items():
        uv = project(calib, pt[None])[0][0]
        s_on = _luma_near(on.render_volume(cid, sim)[1], uv)
        s_off = _luma_near(off.render_volume(cid, sim)[1], uv)
        record_property(f"emission_cam{cid}", f"with={s_on:.4g} without={s_off:.4g}")
        assert s_on > 0.0
        assert s_off <= 1e-3 * s_on


def test_flame_in_state_digest(snap, table):
    sim = make(snap, table)
    sim.ignite(paper_top(snap), 0.03)
    sim.step(20)
    st = sim.state_host()
    d0 = st.digest()
    assert "flame" in st._ARRAYS
    st.flame[0, 0, 0] += 1.0
    assert st.digest() != d0
    sim2 = make(snap, table)
    sim2.ignite(paper_top(snap), 0.03)
    sim2.step(20)
    assert sim2.state_host().digest() == d0  # replay equal with flame included
