"""Degradation controller and ladder application (P0-010 fix round 1)."""

from livefire.app.degradation import DegradationController
from livefire.app.hud import HudInfo, hud_lines, make_ladder
from livefire.app.lifecycle import State
from livefire.app.runner import RunConfig, Runner


def test_controller_walks_ladder():
    c = DegradationController(make_ladder(0.35, 1.0), 0.9, 30)
    got = [c.update(0.5) for _ in range(29)]
    assert got == [None] * 29
    assert c.update(1.0) is None and c.low == 0  # a good frame resets the counter
    assert [c.update(0.5) for _ in range(30)][-1] == 1
    assert [c.update(0.5) for _ in range(30)][-1] == 2
    assert all(c.update(0.1) is None for _ in range(100)) and c.level == 2  # top of the ladder


def test_ladder_applied_logged_recorded_and_in_hud(tmp_path):
    r = Runner(RunConfig(out=tmp_path, grid=16))
    r.setup()
    r.life.transition(State.SIMULATING)
    base_scale, base_step = r.comp.scale, float(r.comp.p["step_factor"])
    seen = []
    for _ in range(60):
        lvl = r.deg.update(0.3)
        if lvl is not None:
            r.apply_level(lvl, "rt 0.3 < 0.9 for 30 frames")
            seen.append(lvl)
            info = HudInfo(r.life.label, 1.0, 1.0, 0.3, 0.0, r.ladder[lvl], {0: 0.0}, 1, False)
            text = " ".join(hud_lines(info))
            assert f"L{lvl}" in text and "DEGRADED" in text
    assert seen == [1, 2]
    log = r.degradation_log
    assert [e["level"] for e in log] == [1, 2]
    assert log[0]["volume_scale"] == 0.35 < base_scale and log[0]["step_factor"] == base_step
    assert log[1]["step_factor"] == 1.0 > base_step
    assert r.life.degraded
    off = " ".join(hud_lines(HudInfo("SIMULATING", 1, 1, 1, 0, r.ladder[0], {0: 0.0}, 1, True)))
    assert "ladder off" in off and "L0 (nominal)" not in off
