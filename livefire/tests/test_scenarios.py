"""Scenarios, snapshot-from-primitives builder, M9 checker, uniform table (P2-003)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from livefire.common.config import load_yaml
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import THICK_DELTA, snapshot_from_primitives
from livefire.scene.synthetic import Box, Rect
from livefire.sim.materials import MaterialTable
from livefire.sim.scenarios import (
    monotonicity_violations,
    ordered_pairs,
    random_scenarios,
    scenario_grid_cfg,
    scripted_scenarios,
    uniform_table,
    unordered_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_yaml("scenarios")


@pytest.fixture(scope="module")
def table():
    return MaterialTable.load()


def _scene():
    sheet = Rect(0, "sheet", "paper", (0.5,) * 3, 1, 0.0, (-0.06, 0.05), (0.06, 0.17), two_sided=False, normal_sign=1.0)
    block = Box(1, "block", "wood", (0.5,) * 3, (-0.05, 0.05, 0.0), (0.05, 0.12, 0.08))
    return [sheet, block]


def test_primitives_snapshot_is_valid(table):
    """Contract checks pass (SceneSnapshot validates on construction), neighbours are symmetric, only thick materials
    make solids, element spacing within 20 % of ``surfel_spacing``, binned area close to the analytic area."""
    grid = scenario_grid_cfg(CFG)
    snap = snapshot_from_primitives(_scene(), grid, seed=3)
    assert snap.cameras == {} and snap.view_depth == {} and snap.calib_version == 0
    nb = snap.neighbors
    for i in range(len(nb)):
        for j in nb[i][nb[i] >= 0]:
            assert i in nb[j]
    np.testing.assert_allclose(np.linalg.norm(snap.normal, axis=1), 1.0, atol=1e-5)
    h, G = snap.h_gas, snap.solid_mask.shape[0]
    cells = lambda m: {tuple(c) for c in np.clip(np.floor((snap.pos[m] - snap.roi.min) / h).astype(int), 0, G - 1)}
    sheet, block = snap.obj_id == 0, snap.obj_id == 1
    assert table.delta[snap.material_id[sheet][0]] < THICK_DELTA <= table.delta[snap.material_id[block][0]]
    assert {tuple(c) for c in np.argwhere(snap.solid_mask)} == cells(block)  # solids = thick object only
    from scipy.spatial import cKDTree

    p = snap.pos[sheet].astype(float)
    d, _ = cKDTree(p).query(p, k=2)
    assert abs(np.median(d[:, 1]) - grid["surfel_spacing"]) <= 0.2 * grid["surfel_spacing"]
    assert snap.area[sheet].sum() == pytest.approx(0.12 * 0.12, rel=0.1)  # edge cells lose a little area


def test_primitives_snapshot_reproducible_and_checks_inputs():
    grid = scenario_grid_cfg(CFG)
    a = snapshot_from_primitives(_scene(), grid, seed=3)
    b = snapshot_from_primitives(_scene(), grid, seed=3)
    np.testing.assert_array_equal(a.pos, b.pos)
    with pytest.raises(ValueError):
        snapshot_from_primitives(_scene(), grid, seed=3, samples_per_cell=2)
    far = Box(2, "far", "wood", (0.5,) * 3, (5.0, 5.0, 5.0), (5.1, 5.1, 5.1))
    with pytest.raises(ValueError, match="no surface elements"):
        snapshot_from_primitives([*_scene(), far], grid, seed=3)


def test_random_scenarios_reproducible_and_disjoint():
    d1, d2 = random_scenarios(CFG, "dev"), random_scenarios(CFG, "dev")
    h = random_scenarios(CFG, "heldout")
    assert len(d1) == CFG["random"]["n_dev"] and len(h) == CFG["random"]["n_heldout"]
    for a, b in zip(d1, d2, strict=True):
        assert a.seed == b.seed and a.meta == b.meta and a.wind_speed == b.wind_speed
    assert {s.seed for s in d1}.isdisjoint({s.seed for s in h})
    assert not {round(s.meta["gap_m"], 9) for s in d1} & {round(s.meta["gap_m"], 9) for s in h}
    r = CFG["random"]
    for s in d1 + h:
        assert r["gap_m"][0] <= s.meta["gap_m"] <= r["gap_m"][1]
        assert r["angle_deg"][0] <= s.meta["angle_deg"] <= r["angle_deg"][1]
        assert r["wind_m_s"][0] <= s.wind_speed <= r["wind_m_s"][1]
        assert r["size_m"][0] <= s.meta["size_target_m"] <= r["size_m"][1]
    names = [s.name for s in scripted_scenarios(CFG)]
    assert names == ["demo_chain", "two_sheets_parallel", "two_sheets_large", "sheet_over_block"]


def test_ordered_pairs_current_table(table):
    pairs = ordered_pairs(table)
    assert sorted(pairs) == sorted(
        [("paper", "cloth"), ("paper", "wood"), ("paper", "plastic"), ("cloth", "wood"), ("cloth", "plastic")]
    )
    assert ("wood", "plastic") not in pairs and ("plastic", "wood") not in pairs
    assert unordered_pairs(table) == [("wood", "plastic")]


def test_ordered_pairs_follow_the_table():
    """Lowering every plastic parameter below wood's makes (plastic, wood) appear."""
    from livefire.common.config import config_dir, load_yaml_text

    ent = load_yaml_text((config_dir() / "materials.yaml").read_text())
    w = ent["wood"]
    ent["plastic"].update(T_ign=w["T_ign"] - 10, rho=w["rho"] / 2, c=w["c"] / 2, delta=w["delta"] / 2, t_delay=w["t_delay"] / 2,
                          k=w["k"] / 2, eps=w["eps"])  # fmt: skip
    t = MaterialTable(ent)
    assert ("plastic", "wood") in ordered_pairs(t)
    assert ("wood", "plastic") not in ordered_pairs(t)


def test_checker_flags_violation_and_none_is_inf(table):
    pairs = ordered_pairs(table)
    dt = 1 / 60
    ok = {"paper": 3.0, "cloth": 4.0, "wood": None, "plastic": None}
    assert monotonicity_violations(ok, pairs, dt) == []
    bad = {"paper": 5.0, "cloth": 4.0, "wood": None, "plastic": None}
    assert monotonicity_violations(bad, pairs, dt) == [("paper", "cloth")]
    inf_first = {"paper": None, "cloth": 4.0, "wood": 6.0, "plastic": None}  # A never, B ignites: violation
    assert set(monotonicity_violations(inf_first, pairs, dt)) == {("paper", "cloth"), ("paper", "wood")}
    tie = {"paper": 4.0 + dt, "cloth": 4.0, "wood": None, "plastic": None}  # within dt_sim: not a violation
    assert monotonicity_violations(tie, pairs, dt) == []
    assert monotonicity_violations({"paper": None, "cloth": None}, pairs, dt) == []  # both never


def test_uniform_table(table):
    ut, uid = uniform_table()
    assert list(ut.names) == list(MATERIAL_CLASSES) and uid == MATERIAL_CLASSES.index("paper")
    fl = [MATERIAL_CLASSES.index(m) for m in ("paper", "cloth", "wood", "plastic")]
    for f in ("T_ign", "t_delay", "mdot_ref", "dH", "chi", "y_soot", "rho", "c", "k", "delta", "eps", "m0"):
        mean = float(np.mean(getattr(table, f)[fl]))
        for i in fl:
            assert getattr(ut, f)[i] == pytest.approx(mean)
    assert ut.flammable[fl].all() and not ut.consumed[fl].any()
    for i in (MATERIAL_CLASSES.index(m) for m in ("metal", "glass", "ceramic", "other_nonflammable")):
        assert not ut.flammable[i] and ut.k[i] == table.k[i]


def test_scenario_runs_and_a_close_paper_target_ignites(table):
    """Two parallel sheets 2 cm apart, hot gas ignored: the coupled run ends at the target's ignition."""
    from livefire.sim.scenarios import Scenario, run_scenario, sim_params_with

    cfg = {**CFG, "t_max_s": 4.0, "grid": {**CFG["grid"], "G": 16}}
    e = Rect(0, "e", "paper", (0.5,) * 3, 1, 0.0, (-0.06, 0.1), (0.06, 0.22), two_sided=False, normal_sign=1.0)
    t = Rect(1, "t", "paper", (0.5,) * 3, 1, 0.02, (-0.06, 0.1), (0.06, 0.22), two_sided=False, normal_sign=-1.0)
    sc = Scenario("tiny", [e, t], (0.0, 0.0, 0.16), 0.08, 0.0, 1, seed=1)
    r = run_scenario(sc, "paper", sim_params_with(), {**load_yaml("app")}, cfg, table)
    assert r["n_lit"] > 0 and r["steps"] > 0 and set(r["burnt_fraction"]) == {"e", "t"}
    assert r["end"] in ("ignited", "t_max", "extinct")
    assert r["ignited"] == (r["t_ignite_s"] is not None)


@pytest.mark.slow
def test_m9_quick_runs(tmp_path):
    """eval/p2_m9_spread.py --quick (1 dev scenario, 2 materials, t_max 5 s, G 16) writes both CSVs."""
    out = tmp_path / "m9"
    r = subprocess.run([sys.executable, str(ROOT / "eval" / "p2_m9_spread.py"), "--quick", "--out-dir", str(out)],
                       check=False, capture_output=True, text=True, timeout=600, cwd=ROOT)  # fmt: skip
    assert r.returncode in (0, 1), r.stderr[-2000:]
    files = sorted(p.name for p in out.glob("*.csv"))
    assert any(f.startswith("quick_m9_spread_") for f in files) and any(f.startswith("quick_m9_summary_") for f in files)


def test_ignition_patch_never_lights_the_target(table):
    """A large ignition radius lights the emitter object only, never the target 2 cm away (obj_id == emitter_obj)."""
    from livefire.sim.coupled import FireSimulation
    from livefire.sim.scenarios import Scenario, ignite_object_patch, sim_params_with

    cfg = {**CFG, "grid": {**CFG["grid"], "G": 16}}
    e = Rect(0, "e", "paper", (0.5,) * 3, 1, 0.0, (-0.06, 0.1), (0.06, 0.22), two_sided=False, normal_sign=1.0)
    t = Rect(1, "t", "paper", (0.5,) * 3, 1, 0.02, (-0.06, 0.1), (0.06, 0.22), two_sided=False, normal_sign=-1.0)
    sc = Scenario("tiny", [e, t], (0.0, 0.0, 0.16), 1.0, 0.0, 1, seed=1)
    snap = snapshot_from_primitives(sc.objects, scenario_grid_cfg(cfg), sc.seed, table)
    sim = FireSimulation(snap, table, sim_params_with(), load_yaml("app"))
    n = ignite_object_patch(sim, snap, sc)
    b = sim.surface.burning.numpy() > 0
    assert n == int(b.sum()) == int((snap.obj_id == 0).sum()) and not b[snap.obj_id == 1].any()
