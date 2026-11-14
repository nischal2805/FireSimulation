"""Click picking on the frozen snapshot (P1R5): pixel -> world, object pick, ignition (P1-002). Headless."""

import cv2
import numpy as np
import pytest

from livefire.app.click import pick_object, pixel_to_world
from livefire.app.interactive import InteractiveController
from livefire.app.lifecycle import Lifecycle, State
from livefire.common.config import load_params, load_yaml
from livefire.scene.builder import ViewInput, build_snapshot, manual_material
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

PAPER, METAL = 3, 2


@pytest.fixture(scope="module")
def world():
    scene, cams = build_table_v1(), load_cameras()
    renders = {c: render_view(scene, cam, seed=0) for c, cam in cams.items()}
    views = {c: ViewInput(cams[c], r.depth, r.valid, r.obj_id, r.rgb) for c, r in renders.items()}
    tab = scene.objects_table()
    snap = build_snapshot(views, {o: manual_material(m) for o, (_, m) in tab.items()},
                          {**load_yaml("grid"), "G": 32}, names={o: nm for o, (nm, _) in tab.items()})  # fmt: skip
    return scene, snap, renders


def make_ctl(snap):
    sim = FireSimulation(snap, MaterialTable.load(), load_params("sim"), load_yaml("app"))
    life = Lifecycle()
    for s in (State.CALIBRATING, State.ACQUIRING, State.READY):
        life.transition(s)
    ctl = InteractiveController(sim, snap, sim.materials, life, load_params("sim"))
    ctl.start()
    return ctl


def interior_pixels(r, obj, n=12):
    """Pixels of ``obj`` whose 5x5 neighbourhood is entirely ``obj`` (away from edges)."""
    m = (r.obj_id == obj) & r.valid
    inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    vs, us = np.nonzero(inner)
    assert len(us) >= n, f"object {obj} too small in view"
    idx = np.linspace(0, len(us) - 1, n).astype(int)
    return [(int(us[i]), int(vs[i])) for i in idx]


@pytest.mark.parametrize("cam", [0, 1])
def test_paper_click_world_point_within_1cm(world, cam):
    scene, snap, renders = world
    for u, v in interior_pixels(renders[cam], PAPER):
        p, reason = pixel_to_world(snap, cam, u, v)
        assert reason == "ok" and p is not None
        assert scene.distance_to_object(PAPER, p[None])[0] < 0.01
        assert pick_object(snap, cam, u, v) == PAPER


@pytest.mark.parametrize("cam", [0, 1])
def test_click_on_paper_ignites_after_one_step(world, cam):
    _, snap, renders = world
    ctl = make_ctl(snap)
    u, v = interior_pixels(renders[cam], PAPER)[6]
    res = ctl.click(cam, u, v, "left")
    assert res.reason == "ignited" and res.n_lit > 0
    ev = ctl.events[-1]
    assert (ev.kind, ev.cam, ev.u, ev.v, ev.step) == ("ignite", cam, u, v, 0)
    assert ev.point is not None and ev.radius == pytest.approx(ctl.radius)
    ctl.sim.step(1)
    st = ctl.sim.state_host()
    burning = st.burning.astype(bool)
    assert burning.any()
    assert set(snap.obj_id[burning]) == {PAPER}


def test_click_on_hole_does_nothing(world):
    _, snap, renders = world
    ctl = make_ctl(snap)
    for cam in (0, 1):  # room wall pixels: valid depth but no scene element -> reported as hole
        vs, us = np.nonzero(renders[cam].obj_id == -1)
        res = ctl.click(cam, int(us[len(us) // 2]), int(vs[len(us) // 2]), "left")
        assert res.reason == "hole" and "hole" in res.message and res.event is None
        assert pick_object(snap, cam, int(us[0]), int(vs[0])) == -1
        vs, us = np.nonzero(snap.view_hole_mask[cam])
        assert len(us)
        res = ctl.click(cam, int(us[0]), int(vs[0]), "left")
        assert res.reason == "hole" and "hole" in res.message and res.event is None
    assert ctl.events == [] and ctl.sim.step_count == 0
    assert not ctl.sim.state_host().burning.any()
    assert ctl.click(0, -5, 10).reason == "outside" and ctl.click(0, 10, 99999).reason == "outside"
    assert ctl.click(7, 10, 10).reason == "no_depth"


def test_click_on_metal_not_flammable(world):
    _, snap, renders = world
    ctl = make_ctl(snap)
    cam = 0
    pix = interior_pixels(renders[cam], METAL, n=1)
    res = ctl.click(cam, *pix[0], "left")
    assert res.reason == "not flammable" and "not flammable" in res.message and res.event is None
    assert ctl.events == [] and not ctl.sim.state_host().burning.any()


def test_edge_pixel_depth_is_median_of_valid_neighbours(world):
    _, snap, renders = world
    r = renders[0]
    m = (r.obj_id == PAPER) & r.valid
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vs, us = np.nonzero(edge)
    u, v = int(us[0]), int(vs[0])
    p, reason = pixel_to_world(snap, 0, u, v)
    assert reason == "ok" and np.isfinite(p).all()
