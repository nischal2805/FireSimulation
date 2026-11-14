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


def _edge_pixels(r, obj):
    m = (r.obj_id == obj) & r.valid
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vs, us = np.nonzero(edge)
    return list(zip(us.tolist(), vs.tolist()))


def test_silhouette_clicks_pick_the_surface_actually_hit(world, capsys):
    """1-px inner boundary pixels of every object, both views. The picked object is "none" (-1) or lies on the
    back-projected point (analytic distance < 1 cm; contact lines such as wood/table or cloth/table can touch).
    Per-object rates against the geometric truth AND against the renderer's obj_id map are printed (pytest -s).
    Note: the renderer's obj_id map disagrees with its own depth at some silhouette pixels (e.g. a pixel labelled
    metal whose depth is the paper behind), so the analytic surface, not the label, is the ground truth."""
    scene, snap, renders = world
    rows = []
    for cam in (0, 1):
        for obj in range(5):
            px = _edge_pixels(renders[cam], obj)
            n = same_label = none = geom_ok = 0
            for u, v in px:
                p, _ = pixel_to_world(snap, cam, u, v)
                if p is None:
                    none += 1
                    continue
                got = pick_object(snap, cam, u, v)
                n += 1
                same_label += got == obj
                none += got == -1
                d = [scene.distance_to_object(o, p[None])[0] for o in range(5)]
                if got == -1:
                    continue
                assert d[got] < 0.01, f"cam{cam} obj{obj} ({u},{v}): picked {got}, {d[got]:.4f} m from the point"
                geom_ok += got == int(np.argmin(d))
            rows.append((cam, obj, len(px), same_label, none, geom_ok))
    with capsys.disabled():
        print("\nsilhouette pick rates: cam obj n_edge_px | == obj_id label | none | == analytic nearest object")
        for c, o, n, sl, no, g in rows:
            print(f"  cam{c} obj{o}: n={n} label={sl / n:.3f} none={no / n:.3f} geom={g / n:.3f}")
    # metal-box boundary clicks never light anything
    ctl = make_ctl(snap)
    for cam in (0, 1):
        px = _edge_pixels(renders[cam], METAL)
        for i in range(0, len(px), max(1, len(px) // 80)):
            res = ctl.click(cam, *px[i], "left")
            if res.reason == "ignited":
                # a boundary click that ignites must have landed within 1 cm of a flammable surface
                assert min(scene.distance_to_object(o, res.point[None])[0] for o in (1, 3, 4)) < 0.01


def test_click_records_pixel_used(world):
    _, snap, renders = world
    from livefire.app.click import resolve_click

    u, v = interior_pixels(renders[0], PAPER, 1)[0]
    r = resolve_click(snap, 0, u + 0.3, v - 0.2)
    assert r.reason == "ok" and r.pixel == (u, v) and r.depth == pytest.approx(float(snap.view_depth[0][v, u]), abs=0.02)
    ctl = make_ctl(snap)
    res = ctl.click(0, u, v)
    assert res.pixel == (u, v) and res.depth > 0


def test_clicks_gated_on_lifecycle(world):
    _, snap, renders = world
    ctl = make_ctl(snap)
    u, v = interior_pixels(renders[0], PAPER, 1)[0]
    ctl.key("r")  # -> ACQUIRING
    for button in ("left", "right"):
        res = ctl.click(0, u, v, button)
        assert res.reason == "not ready" and res.message == "not ready" and res.event is None
    assert ctl.events == [] and ctl.selected is None
    ctl.finish_reacquire("ok")
    assert ctl.click(0, u, v).reason == "ignited"
