"""Gas solver baseline (P0-008, PRD 8.4): MAC projection quality, boundaries, symmetry, buoyancy, sources."""

import copy
from itertools import pairwise

import numpy as np
import pytest
import warp as wp

from livefire.common.config import Params, load_params
from livefire.sim.device import select_device
from livefire.sim.gas import GasSolver

DEV = select_device()
G = 32
H = 3.0 / G
ROI_MIN = np.array([-1.5, -1.5, -0.1])
DT = 1.0 / 60.0


def params(**over: float) -> Params:
    """configs/sim.yaml with value overrides; ranges widened to include the override."""
    p = load_params("sim")
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return Params(e)


def solver(solid: np.ndarray | None = None, g: int = G, **over: float) -> GasSolver:
    s = np.zeros((g, g, g), np.uint8) if solid is None else solid
    return GasSolver(g, 3.0 / g, ROI_MIN, s, params(**over), DEV, seed=7)


def random_velocity(s: GasSolver, seed: int = 0) -> None:
    r = np.random.default_rng(seed)
    g = s.G
    s.set_velocity(
        r.standard_normal((g + 1, g, g)), r.standard_normal((g, g + 1, g)), r.standard_normal((g, g, g + 1))
    )


def blob(g: int, centre: tuple[float, float, float], sigma: float, amp: float, t_amb: float = 293.15) -> np.ndarray:
    """T field with a Gaussian blob; ``centre`` in cell units (cell i centre at i + 0.5)."""
    i = np.arange(g) + 0.5
    x, y, z = np.meshgrid(i - centre[0], i - centre[1], i - centre[2], indexing="ij")
    return (t_amb + amp * np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))).astype(np.float32)


def com(T: np.ndarray, t_amb: float = 293.15) -> np.ndarray:
    """Centre of mass of T - T_amb in cell units (cell i centre at i + 0.5)."""
    f = T - t_amb
    i = np.arange(T.shape[0]) + 0.5
    return np.array([(f.sum((1, 2)) * i).sum(), (f.sum((0, 2)) * i).sum(), (f.sum((0, 1)) * i).sum()]) / f.sum()


def test_projection_reduces_divergence_10x_default_iters(record_property):
    s = solver()
    random_velocity(s)
    before, after = s.project(DT)  # default n_iters
    record_property("divergence_before", before)
    record_property("divergence_after", after)
    record_property("reduction_factor", before / after)
    print(f"\nprojection G=32 n_iters={int(params()['n_iters'])}: ||div|| {before:.4g} -> {after:.4g} ({before / after:.1f}x)")
    assert before / after >= 10.0


def test_projection_factor_vs_iterations():
    """Reports the measured reduction for several n_iters (more iterations never worse)."""
    factors = []
    for n in (5, 10, 20, 60):
        s = solver()
        random_velocity(s)
        b, a = s.project(DT, n)
        factors.append(b / a)
        print(f"\nn_iters={n}: {b / a:.1f}x")
    assert all(x < y for x, y in pairwise(factors))


def test_no_flux_through_solids_walls_after_projection():
    rng = np.random.default_rng(3)
    solid = np.zeros((G, G, G), np.uint8)
    solid[10:16, 12:20, 0:6] = 1
    solid[20:24, 4:8, 8:14] = 1
    solid[rng.random((G, G, G)) < 0.02] = 1
    s = solver(solid)
    random_velocity(s, 1)
    b, a = s.project(DT)
    assert a < b / 10
    u, v, w = s.u.numpy(), s.v.numpy(), s.w.numpy()
    # domain walls (the top z face is open)
    assert np.all(u[0] == 0) and np.all(u[G] == 0)
    assert np.all(v[:, 0] == 0) and np.all(v[:, G] == 0)
    assert np.all(w[:, :, 0] == 0)
    # faces touching a solid cell
    sb = solid.astype(bool)
    assert np.all(u[1:G][sb[:-1] | sb[1:]] == 0)
    assert np.all(v[:, 1:G][sb[:, :-1] | sb[:, 1:]] == 0)
    assert np.all(w[:, :, 1:G][sb[:, :, :-1] | sb[:, :, 1:]] == 0)
    assert np.all(w[:, :, G][sb[:, :, G - 1]] == 0)
    # top face is open: outflow allowed
    assert np.abs(w[:, :, G][~sb[:, :, G - 1]]).max() > 0


def test_divergence_measured_over_fluid_cells_only():
    solid = np.zeros((G, G, G), np.uint8)
    solid[8:24, 8:24, 0:4] = 1
    s = solver(solid)
    random_velocity(s, 2)
    b, a = s.project(DT)
    assert np.isfinite(b) and np.isfinite(a) and a < b / 10


def test_zero_wind_symmetry():
    s = solver(eps_vc=0.3)
    s.set_scalars(T_g=blob(G, (G / 2, G / 2, 8.0), 2.0, 300.0))
    c0 = com(s.T_g.numpy())
    for _ in range(60):
        s.step(DT)
    T = s.T_g.numpy()
    assert T.max() > 293.15 + 1.0, "blob vanished; test would be vacuous"
    c1 = com(T)
    drift = np.abs(c1[:2] - c0[:2])
    print(f"\nsymmetry: COM drift (cells) x={drift[0]:.3g} y={drift[1]:.3g}; z {c0[2]:.2f} -> {c1[2]:.2f}")
    assert np.all(drift < 0.25)
    f = T - 293.15
    mirror = f[::-1]
    rel = np.linalg.norm(f - mirror) / np.linalg.norm(f)
    mirror_y = np.linalg.norm(f - f[:, ::-1]) / np.linalg.norm(f)
    print(f"mirror-asymmetry rel L2 x={rel:.3g} y={mirror_y:.3g}")
    assert rel < 1e-3
    assert mirror_y < 1e-3


def test_buoyancy_blob_rises():
    s = solver()
    s.set_scalars(T_g=blob(G, (G / 2, G / 2, 6.0), 2.0, 300.0))
    z0 = com(s.T_g.numpy())[2]
    for _ in range(30):
        s.step(DT)
    z1 = com(s.T_g.numpy())[2]
    assert z1 > z0 + 0.3
    w = s.velocity_cell_centered()[..., 2]
    assert w.max() > 0.5


def test_quiescent_ambient_stays_quiescent():
    s = solver()
    for _ in range(10):
        s.step(DT)
    assert np.abs(s.velocity_cell_centered()).max() < 1e-5
    assert np.abs(s.T_g.numpy() - 293.15).max() < 1e-4


def test_cell_centred_velocity_accessor_shape_and_values():
    s = solver()
    u = np.full((G + 1, G, G), 2.0, np.float32)
    s.set_velocity(u, np.zeros((G, G + 1, G)), np.zeros((G, G, G + 1)))
    v = s.velocity_cell_centered()
    assert v.shape == (G, G, G, 3) and v.dtype == np.float32
    np.testing.assert_allclose(v[..., 0], 2.0)
    assert np.all(v[..., 1:] == 0)


def test_velocity_clamp_counted_and_effective():
    """The clamp acts before projection (PRD 8.4 order), so projection may push a component past it; the
    clamped run must still be far below the unclamped one, and violations are counted."""

    def run(cfl: float) -> tuple[float, int]:
        s = solver(cfl_max=cfl)
        r = np.random.default_rng(5)
        s.set_velocity(*(50.0 * r.standard_normal(sh) for sh in ((G + 1, G, G), (G, G + 1, G), (G, G, G + 1))))
        s.step(DT)
        return max(float(np.abs(a.numpy()).max()) for a in (s.u, s.v, s.w)), s.counters()["vel_clamp"]

    m_clamped, n_clamped = run(1.0)
    m_free, n_free = run(10.0)
    vmax = 1.0 * H / DT
    print(f"\nmax |face vel|: clamped {m_clamped:.2f} (clamp {vmax:.2f}, {n_clamped} events), cfl 10 {m_free:.2f} ({n_free})")
    assert n_clamped > n_free
    assert m_clamped < 0.5 * m_free


def test_temperature_clamp_counted():
    s = solver(T_g_max=1000.0)
    s.set_scalars(T_g=np.full((G, G, G), 1900.0, np.float32))
    s.step(DT)
    assert s.counters()["T_g_clamp"] > 0
    assert s.T_g.numpy().max() <= 1000.0


def test_cooling_and_soot_decay():
    s = solver()
    s.set_scalars(T_g=np.full((G, G, G), 393.15, np.float32), soot=np.full((G, G, G), 1.0, np.float32))
    s.step(DT)
    k_cool, k_soot = 1.5, 0.3
    np.testing.assert_allclose(s.T_g.numpy(), 293.15 + 100.0 * np.exp(-k_cool * DT), rtol=1e-5)
    np.testing.assert_allclose(s.soot.numpy(), np.exp(-k_soot * DT), rtol=1e-4)


@pytest.mark.parametrize("with_solid", [False, True])
def test_source_injection_conserves_energy_and_soot(with_solid):
    solid = np.zeros((G, G, G), np.uint8)
    if with_solid:
        solid[16, 16, 6:12] = 1  # point sits next to this column
    s = solver(solid, k_cool=0.0, k_soot=0.0)
    pt = np.array([[0.02, 0.03, 0.07 + 0.2]], np.float32)
    hw = wp.array(np.array([5000.0], np.float32), dtype=wp.float32, device=DEV)
    sk = wp.array(np.array([1e-4], np.float32), dtype=wp.float32, device=DEV)
    s.step(DT, wp.array(pt, dtype=wp.float32, device=DEV), hw, sk)
    p = params()
    cell_vol = H**3
    dT = s.T_g.numpy().astype(np.float64) - float(np.float32(293.15))  # field is float32
    # advection/projection of a uniform ambient field leaves it unchanged, so the sum is the injected energy
    e = dT.sum() * p["rho_g"] * p["c_p"] * cell_vol
    assert e == pytest.approx(5000.0 * DT, rel=1e-4)
    assert s.soot.numpy().astype(np.float64).sum() * cell_vol == pytest.approx(1e-4 * DT, rel=1e-4)
    assert dT[solid.astype(bool)].max() == 0.0 if with_solid else True
    assert s.counters()["src_lost"] == 0


def test_source_outside_roi_is_counted_not_deposited():
    s = solver(k_cool=0.0)
    pt = wp.array(np.array([[9.0, 0.0, 1.0]], np.float32), dtype=wp.float32, device=DEV)
    hw = wp.array(np.array([1000.0], np.float32), dtype=wp.float32, device=DEV)
    sk = wp.array(np.array([0.0], np.float32), dtype=wp.float32, device=DEV)
    s.step(DT, pt, hw, sk)
    assert s.counters()["src_lost"] == 1
    assert np.abs(s.T_g.numpy() - 293.15).max() < 1e-4


def test_sample_T_trilinear():
    s = solver()
    i = np.arange(G) + 0.5
    ramp = (293.15 + 10.0 * i)[:, None, None] * np.ones((1, G, G))
    s.set_scalars(T_g=ramp.astype(np.float32))
    x = ROI_MIN[0] + 7.25 * H  # cell-unit coordinate 7.25 -> between centres 6.5 and 7.5
    pts = wp.array(np.array([[x, 0.1, 0.5]], np.float32), dtype=wp.float32, device=DEV)
    out = wp.zeros(1, dtype=wp.float32, device=DEV)
    s.sample_T(pts, out)
    assert out.numpy()[0] == pytest.approx(293.15 + 10.0 * 7.25, abs=1e-2)


def test_wind_inflow_seeded_and_default_off():
    a = solver(wind_speed=1.0, wind_noise=0.2)
    b = solver(wind_speed=1.0, wind_noise=0.2)
    for _ in range(5):
        a.step(DT)
        b.step(DT)
    ua = a.u.numpy()
    np.testing.assert_array_equal(ua, b.u.numpy())  # same seed -> identical
    assert ua[0].mean() == pytest.approx(1.0, abs=0.1)  # inflow on -x wall
    assert a.velocity_cell_centered()[..., 0].mean() > 0.1
    z = solver()
    z.step(DT)
    assert np.all(z.u.numpy()[0] == 0)


def test_odd_grid_rejected():
    with pytest.raises(ValueError):
        solver(g=7)
