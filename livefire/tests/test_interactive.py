"""Interactive controller: selection, overrides, keys, recording/replay, window layout (P1-002). Headless."""

import numpy as np
import pytest

from livefire.app.interactive import RADIUS_STEP, InteractiveController, ViewLayout
from livefire.app.lifecycle import State
from livefire.app.replay import read_events, replay
from livefire.app.window import InteractiveSession
from livefire.common.config import load_params, load_yaml


@pytest.fixture(scope="module")
def sess():
    return InteractiveSession("table_v1", grid=32)


@pytest.fixture()
def ctl(sess):
    c = InteractiveController(sess.controller.sim.__class__(sess.snapshot, sess.controller.materials,
                                                            load_params("sim"), load_yaml("app")),
                              sess.snapshot, sess.controller.materials, _ready_life(), load_params("sim"))
    c.start()
    return c


def _ready_life():
    from livefire.app.lifecycle import Lifecycle

    life = Lifecycle()
    for s in (State.CALIBRATING, State.ACQUIRING, State.READY):
        life.transition(s)
    return life


def _pix(snap_obj_pos, ctl, cam, obj):
    """A pixel of ``obj`` seen (unoccluded) in ``cam``: projected element centres that pick back to ``obj``."""
    from livefire.app.click import pick_object
    from livefire.common.geometry import project

    pos = ctl.snap.pos[ctl.snap.obj_id == obj].astype(np.float64)
    uv, _, ok = project(ctl.snap.cameras[cam], pos)
    for u, v in uv[ok][::7]:
        if pick_object(ctl.snap, cam, float(u), float(v)) == obj:
            return float(u), float(v)
    raise AssertionError(f"object {obj} not visible in camera {cam}")


def test_right_click_select_then_key_overrides_material(ctl):
    u, v = _pix(None, ctl, 0, 2)  # metal box
    res = ctl.click(0, u, v, "right")
    assert res.reason == "selected" and ctl.selected == 2 and "metal_box" in res.message
    assert any("REQUIRES REVIEW" in ln or "reviewed" in ln for ln in ctl.hud_lines())
    assert "metal" in ctl.hud_lines()[0]
    ctl.key("3")  # wood
    assert ctl.overrides == {2: "wood"} and int(np.argmax(ctl.snap.objects[2].material.class_probabilities)) == 4  # frozen registry
    ov = ctl.events[-1]
    assert (ov.kind, ov.obj_id, ov.material) == ("override", 2, "wood")
    assert "wood" in ctl.hud_lines()[0] and "reviewed" in ctl.hud_lines()[0]
    # the box is now flammable: a left click on it ignites
    res = ctl.click(0, u, v, "left")
    assert res.reason == "ignited" and res.n_lit > 0


def test_key_without_selection_and_right_click_on_hole(ctl):
    assert "select" in ctl.key("4") and ctl.events == []
    assert ctl.click(0, 5, 5, "right").reason == "no object"  # top-left: background wall


def test_pause_radius_overlay_view_keys(ctl):
    t0 = ctl.sim.step_count
    assert ctl.tick(0.1) is not None and ctl.sim.step_count > t0
    ctl.key("space")
    assert ctl.life.state is State.PAUSED
    n = ctl.sim.step_count
    assert ctl.tick(0.1) is None and ctl.sim.step_count == n
    ctl.key("space")
    assert ctl.life.state is State.SIMULATING
    r0 = ctl.radius
    ctl.key("+")
    assert ctl.radius == pytest.approx(r0 * RADIUS_STEP)
    for _ in range(60):
        ctl.key("+")
    assert ctl.radius == ctl.r_max == 0.2
    for _ in range(120):
        ctl.key("-")
    assert ctl.radius == ctl.r_min == 0.005
    ctl.key("o")
    assert ctl.overlay
    assert len(ctl.visible_cams()) == 2
    ctl.key("v")
    assert ctl.visible_cams() == [0]
    ctl.key("v")
    assert ctl.visible_cams() == [1]
    ctl.key("v")
    assert len(ctl.visible_cams()) == 2
    ctl.key("s")
    assert ctl.screenshot_requested
    assert not ctl.quit
    ctl.key("q")
    assert ctl.quit


def test_reacquire_request(ctl):
    ctl.key("r")
    assert ctl.life.state is State.ACQUIRING and ctl.reacquire_requested
    assert ctl.tick(0.1) is None
    ctl.finish_reacquire("done")
    assert ctl.life.state is State.SIMULATING and not ctl.reacquire_requested and ctl.message == "done"


def test_scripted_session_replays_to_same_digest(ctl, tmp_path):
    u, v = _pix(None, ctl, 1, 3)  # paper in view 1
    mu, mv = _pix(None, ctl, 0, 2)
    for _ in range(3):
        ctl.tick(1 / 30)
    assert ctl.click(1, u, v, "left").reason == "ignited"
    for _ in range(4):
        ctl.tick(1 / 30)
    ctl.click(0, mu, mv, "right")
    ctl.key("3")
    ctl.key("+")
    for _ in range(4):
        ctl.tick(1 / 30)
    assert ctl.click(0, mu, mv, "left").reason == "ignited"
    for _ in range(3):
        ctl.tick(1 / 30)
    digest = ctl.save_session(tmp_path, "table_v1", {"G": 32}, load_yaml("app"))
    ev = read_events(tmp_path / "events.jsonl")
    assert [e.kind for e in ev] == ["ignite", "override", "ignite"]
    assert ev[0].cam == 1 and ev[0].u == pytest.approx(u) and ev[0].step > 0
    res = replay(tmp_path)
    assert res["ok"] and res["actual"] == digest


def test_layout_roundtrip_non_integer_scale():
    lay = ViewLayout(1280, 720, 2, 1500, 800)  # scale 1500/2560 = 0.586..., not an integer
    assert lay.scale == pytest.approx(1500 / 2560) and lay.pane_w == 750
    for pane in (0, 1):
        for u, v in [(0.0, 0.0), (640.0, 360.0), (1279.0, 719.0), (123.0, 456.0)]:
            x, y = lay.from_view(pane, u, v)
            xi, yi = round(x), round(y)  # a display pixel
            p2, u2, v2 = lay.to_view(xi, yi)
            assert p2 == pane
            assert abs(u2 - u) <= 0.5 / lay.sx + 1e-6 and abs(v2 - v) <= 0.5 / lay.sy + 1e-6
            # and display -> view -> display is exact
            x3, y3 = lay.from_view(*lay.to_view(xi, yi))
            assert (x3, y3) == pytest.approx((xi, yi))
    assert lay.to_view(lay.pane_w * 2, 0) is None and lay.to_view(-1, 0) is None
    assert lay.to_view(0, lay.pane_h) is None
    assert lay.to_view(lay.pane_w - 1, 0)[0] == 0 and lay.to_view(lay.pane_w, 0)[0] == 1
    assert ViewLayout(1280, 720, 1, 4000, 4000).scale == 1.0  # never upscaled


def test_compose_headless_smoke(sess):
    for src in sess.sources:
        src.start()
    try:
        sess.controller.start()
        sess.controller.key("o")
        lay = sess.layout
        img = sess.compose(1 / 30)
        assert img.dtype == np.uint8 and img.shape[1] == 2 * lay.pane_w and img.shape[0] == lay.pane_h
        sess.controller.key("v")
        assert sess.compose(1 / 30).shape[1] == sess.layout.pane_w
    finally:
        for src in sess.sources:
            src.stop()
