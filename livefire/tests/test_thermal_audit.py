"""Thermal model audit (P2-002, PRD 8.1/8.2, P2R8): closed-form checks of the lumped model, units, parameter table."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
import yaml
from scipy.integrate import solve_ivp

from livefire.common.config import Params, config_dir, load_params, load_yaml
from livefire.common.geometry import AABB
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import snapshot_from_synthetic
from livefire.scene.snapshot import ObjectRegistry, SceneSnapshot
from livefire.sim.device import select_device
from livefire.sim.materials import MaterialTable
from livefire.sim.surface import SIGMA, SurfaceFireModel

DEV = select_device()
MID = {n: i for i, n in enumerate(MATERIAL_CLASSES)}
S = 0.015


def sim_params(**over: float) -> Params:
    """configs/sim.yaml with value overrides; ranges widened to include the override."""
    p = load_params("sim")
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return Params(e)


def isolated(pos, normal, areas, materials) -> SceneSnapshot:
    """Elements without neighbours (no conduction)."""
    n = len(pos)
    return SceneSnapshot(
        version=1, calib_version=0, roi=AABB(np.full(3, -1.0), np.full(3, 1.0)), h_gas=0.03, cameras={},
        pos=np.asarray(pos, np.float32), normal=np.asarray(normal, np.float32), area=np.asarray(areas, np.float32),
        obj_id=np.zeros(n, np.int32), material_id=np.array([MID[m] for m in materials], np.uint8),
        material_conf=np.ones(n, np.float32), neighbors=-np.ones((n, 1), np.int32),
        solid_mask=np.zeros((1, 1, 1), np.uint8), solid_fraction=np.zeros((1, 1, 1), np.float32),
        view_depth={}, view_hole_mask={}, objects=ObjectRegistry(), surfel_spacing=S,
    )  # fmt: skip


def gas(n, t):
    return wp.array(np.full(n, t, np.float32), dtype=wp.float32, device=DEV)


@pytest.mark.parametrize(("receiver", "area_e"), [("paper", 2.0e-3), ("wood", 1.0e-2)])
def test_radiative_lumped_heating_matches_ode(receiver, area_e):
    """Receiver (no neighbours, h_c = 0) facing a burning wood patch 0.05 m away. T_s(t) vs solve_ivp of
    C'' dT/dt = q_abs - eps sigma (T^4 - T_amb^4), q_abs read from compute_radiation(); ignition time = time to
    reach T_ign plus t_delay (E accumulates at dt while T >= T_ign)."""
    table = MaterialTable.load()
    dt = 0.01  # emitter patch area_e chosen per receiver so that T_ign is reached (wood: C'' 5100 J/(m^2 K))
    snap = isolated([[0, 0, 0], [0, 0, 0.05]], [[0, 0, 1], [0, 0, -1]], [area_e, 1e-4], ["wood", receiver])
    p = sim_params(rad_mode=2, h_c=0.0, f_feedback=0.0)
    m = SurfaceFireModel(snap, table, p, DEV)
    assert m.ignite(np.zeros(3), 1e-3) == 1
    q_abs = float(m.compute_radiation().numpy()[1])
    assert q_abs > 0
    i = MID[receiver]
    cpp, eps, t_amb, t_ign, t_delay = (
        table.C_areal[i],
        table.eps[i],
        p["T_amb"],
        table.T_ign[i],
        table.t_delay[i],
    )

    def rhs(_t, y):
        return [(q_abs - eps * SIGMA * (y[0] ** 4 - t_amb**4)) / cpp]

    reach = lambda _t, y: y[0] - t_ign
    reach.terminal = True
    sol = solve_ivp(rhs, [0, 300], [t_amb], rtol=1e-10, atol=1e-8, events=reach, dense_output=True)
    t_cross = float(sol.t_events[0][0])
    t_ode_ign = t_cross + t_delay
    tg = gas(2, t_amb)
    pts = [0.2, 0.4, 0.6, 0.8, 0.95]
    checks = sorted({round(f * t_cross / dt) for f in pts})
    t_sim_ign = None
    for k in range(1, round(1.5 * t_ode_ign / dt) + 2):
        m.step(dt, tg)
        T = m.T_s.numpy()
        if k in checks:
            assert T[1] == pytest.approx(float(sol.sol(k * dt)[0]), rel=0.01), f"t={k * dt:.3f}"
        if t_sim_ign is None and m.burning.numpy()[1]:
            t_sim_ign = k * dt
            break
    assert m.burning.numpy()[0] == 1  # emitter still burning throughout: constant emission
    assert t_sim_ign is not None
    assert abs(t_sim_ign - t_ode_ign) <= 0.02 * t_ode_ign + dt, (t_sim_ign, t_ode_ign)


def test_convective_relaxation_matches_closed_form(tmp_path):
    """h_c = 25, no radiation (rad_mode 0; eps = 1e-12, the validator requires eps > 0), gas fixed at 400 K:
    T_s = T_g + (T_amb - T_g) exp(-h_c t / C'')."""
    raw = yaml.safe_load((config_dir() / "materials.yaml").read_text())
    raw["paper"]["eps"] = 1e-12
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    table = MaterialTable.load(tmp_path / "m.yaml")
    snap = isolated([[0, 0, 0]], [[0, 0, 1]], [1e-4], ["paper"])
    p = sim_params(rad_mode=0, h_c=25.0, f_feedback=0.0)
    m = SurfaceFireModel(snap, table, p, DEV)
    dt, tg, t_amb, tau = 0.01, gas(1, 400.0), p["T_amb"], table.C_areal[0] / 25.0
    marks = {round(t / dt) for t in (1.0, 3.0, 8.0)}
    for k in range(1, max(marks) + 1):
        m.step(dt, tg)
        if k in marks:
            ref = 400.0 + (t_amb - 400.0) * np.exp(-k * dt / tau)
            assert m.T_s.numpy()[0] == pytest.approx(ref, rel=0.01), f"t={k * dt}"
    assert not m.burning.numpy()[0]


def test_units_sanity():
    table = MaterialTable.load()
    np.testing.assert_allclose(table.C_areal, table.rho * table.c * table.delta, rtol=1e-12)
    assert (table.C_areal > 0).all()
    snap = snapshot_from_synthetic("table_v1")
    m = SurfaceFireModel(snap, table, load_params("sim"), DEV)
    assert 0 < m.dt_cond_max <= np.inf and np.isfinite(m.dt_conv_max) and m.dt_conv_max > 0
    n = m.substeps_for(load_yaml("app")["dt_sim"])
    assert isinstance(n, int) and 1 <= n < 10**6


def test_sim_yaml_audit():
    p = load_params("sim")  # validates value/unit/range/status
    assert {"rad_mode", "eps_f", "T_rad_flame", "rad_R", "h_c", "k_flame"} <= set(p.keys())
    for k in p.keys():  # noqa: SIM118
        e = p.meta(k)
        assert isinstance(e["unit"], str) and e["unit"].strip(), f"{k}: empty unit"
        assert e["status"] in {"physical_estimate", "artistic"}, k


def test_radiation_compare_quick_runs(tmp_path):
    """bench/p2_radiation_compare.py --quick writes both CSVs (Appendix C header, effect rows for 3 modes)."""
    root = Path(__file__).resolve().parents[1]
    b, e = tmp_path / "b.csv", tmp_path / "e.csv"
    r = subprocess.run(
        [sys.executable, str(root / "bench" / "p2_radiation_compare.py"), "--quick", "--out-bench", str(b), "--out-effect", str(e)],
        check=False, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    from livefire.common.bench_csv import read_rows

    rows = read_rows(b)
    assert {x["stage"] for x in rows} == {"radiation", "surface_step"}
    assert {x["variant"].split("_")[0] for x in rows} == {"mode0", "mode1", "mode2"}
    assert len(e.read_text().strip().splitlines()) > 3 * 5


def test_default_rad_mode_is_gather():
    """ADR-0006: the default radiation technique is the bounded gather (mode 2); mode 1 is the documented fallback."""
    assert load_params("sim")["rad_mode"] == 2
