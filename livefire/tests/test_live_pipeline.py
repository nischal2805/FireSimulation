"""P1-006: live mode end to end, headless: recorded synthetic clips as cameras, oracle depth, stored synthetic calibration."""

import csv
import gc
import weakref
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from livefire.app.change_monitor import ChangeConfig, ChangeMonitor, footprint_mask
from livefire.app.lifecycle import State
from livefire.app.live import (
    LATENCY_COLUMNS,
    CameraHealth,
    LiveConfig,
    LivePipeline,
    make_depth_factory,
    verify_calibration,
)
from livefire.calib.board import BoardSpec, board_corners_world, detect_charuco
from livefire.calib.store import CalibrationSet, CameraRecord, load_calibration
from livefire.calib.synthetic_board import SyntheticBoardCamera, random_view_poses
from livefire.capture.file_source import VideoFileSource
from livefire.capture.synthetic_source import SyntheticCameraSource
from livefire.capture.types import Frame
from livefire.common.config import load_yaml
from livefire.common.geometry import CameraCalibration, invert_transform
from livefire.fusion.objects import load_acquisition_config
from livefire.render.geometry_overlay import project_elements
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

SCENE = build_table_v1()
CAMS = load_cameras()
TRUTH = {c: render_view(SCENE, CAMS[c]) for c in CAMS}
GRID = {**load_yaml("grid"), "G": 32}
FAST = ChangeConfig(threshold=0.03, frames=3)  # trigger after 3 new frames (production default: configs/live.yaml)


# ---------------------------------------------------------------------------------------------- change monitor (unit)
def _noisy(rgb, rng, sigma=3.0):
    return np.clip(rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape), 0, 255).astype(np.uint8)


def _moved(rgb):
    out = rgb.copy()
    h, w = out.shape[:2]
    out[h // 3 : h // 3 + h // 5, w // 3 : w // 3 + w // 6] = (230, 40, 40)  # a 3 % patch appears (object moved in)
    return out


def test_monitor_ignores_sensor_noise_and_gain():
    rng = np.random.default_rng(0)
    base = TRUTH[0].rgb
    mon = ChangeMonitor(FAST)
    mon.set_reference({0: _noisy(base, rng)})
    for i in range(30):
        gain = 1.0 + 0.06 * (i % 3 - 1)  # auto-exposure wobble +-6 %
        frame = np.clip(_noisy(base, rng, 4.0).astype(np.float32) * gain, 0, 255).astype(np.uint8)
        st = mon.update({0: (i, frame)})
        assert not st.triggered and st.fraction[0] < 0.01, st


def test_monitor_triggers_on_injected_change_after_n_frames():
    rng = np.random.default_rng(1)
    base = TRUTH[0].rgb
    mon = ChangeMonitor(FAST)
    mon.set_reference({0: _noisy(base, rng)})
    for i in range(5):
        assert not mon.update({0: (i, _noisy(base, rng))}).triggered
    seen = []
    for i in range(5, 12):
        seen.append(mon.update({0: (i, _noisy(_moved(base), rng))}).triggered)
    assert seen == [False, False, True, True, True, True, True]  # 3 consecutive new frames
    again = mon.update({0: (11, _noisy(_moved(base), rng))})  # same seq: does not advance the run
    assert again.consecutive == mon.consecutive


def test_monitor_run_resets_and_same_seq_does_not_count():
    base = TRUTH[0].rgb
    mon = ChangeMonitor(FAST)
    mon.set_reference({0: base})
    mon.update({0: (0, _moved(base))})
    mon.update({0: (1, _moved(base))})
    assert mon.consecutive == 2
    mon.update({0: (1, _moved(base))})  # no new frame
    assert mon.consecutive == 2
    mon.update({0: (2, base)})  # back to normal: run resets
    assert mon.consecutive == 0


def test_monitor_excludes_fire_footprint():
    base = TRUTH[0].rgb
    changed = _moved(base)
    cfg = replace(FAST, footprint_dilate_px=0)
    foot = footprint_mask(base, changed, cfg)  # "composite" differs from the input exactly where the patch is
    assert foot.any() and 0.02 < foot.mean() < 0.06
    mon = ChangeMonitor(cfg)
    mon.set_reference({0: base})
    assert mon.changed_fraction(0, changed) > 0.03
    assert mon.changed_fraction(0, changed, foot) < 0.003  # inside the footprint: not a scene change


# ---------------------------------------------------------------------------------------------- camera health (unit)
def test_camera_health_drop_and_restore_with_injected_clock():
    t = [0.0]
    h = CameraHealth(3.0, lambda: t[0])
    assert h.update({0: "connected", 1: "failed"}) == set()
    t[0] = 2.9
    assert h.update({0: "connected", 1: "failed"}) == set()
    t[0] = 3.0
    assert h.update({0: "connected", 1: "failed"}) == {1}
    t[0] = 4.0
    assert h.update({0: "connected", 1: "reconnecting"}) == set()  # anything but failed restores at once
    t[0] = 5.0
    assert h.update({0: "connected", 1: "failed"}) == set()  # the timer restarts
    assert [e[1:] for e in h.events] == [(1, "dropped"), (1, "restored")]


# ---------------------------------------------------------------------------------------------- calibration check
SPEC = BoardSpec()
SIZE = (640, 480)
K_RAW = np.array([[560.0, 0, 322.0], [0, 556.0, 241.0], [0, 0, 1.0]])
DIST = np.array([-0.28, 0.09, 0.0008, -0.0006, 0.0])  # clearly non-zero lens distortion (the synthetic rig has none)


def _board_setup(version=7):
    rng = np.random.default_rng(3)
    K_undist, _ = cv2.getOptimalNewCameraMatrix(K_RAW, DIST, SIZE, 0.0)
    T = random_view_poses(1, rng, distance=(0.5, 0.6))[0]
    raw = SyntheticBoardCamera(SPEC, K_RAW, DIST, SIZE).render(T, rng)  # distorted gray image
    rgb = np.ascontiguousarray(np.repeat(raw[..., None], 3, axis=2))
    cal = CameraCalibration(0, SIZE[0], SIZE[1], K_undist, DIST, T, version)
    calib = CalibrationSet(version, SPEC, {0: CameraRecord(cal, K_RAW, 0.0)}, {})
    return calib, cal, rgb, T


def _rms(cal, spec, img_bgr, T):
    corners, ids = detect_charuco(spec, img_bgr)
    assert len(ids) >= 10
    T_cw = invert_transform(T)
    proj, _ = cv2.projectPoints(board_corners_world(spec)[ids], cv2.Rodrigues(T_cw[:3, :3])[0], T_cw[:3, 3], cal.K, np.zeros(5))
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))


def test_calibration_verify_ok_high_and_no_board():
    calib, cal, rgb, T = _board_setup()
    f = Frame(0, rgb, 1, 0, 0)
    (ok,) = verify_calibration(calib, {0: f}, 2.0)
    assert ok.state == "ok" and ok.rms_px < 1.0, ok
    bad_T = T.copy()
    bad_T[:3, 3] += np.array([0.03, 0.0, 0.0])  # camera "moved" 3 cm since calibration
    bad = CalibrationSet(7, SPEC, {0: CameraRecord(replace(cal, T_world_from_camera=bad_T), K_RAW, 0.0)}, {})
    (hi,) = verify_calibration(bad, {0: f}, 2.0)
    assert hi.state == "high" and hi.rms_px > 2.0 and "recalibrat" in hi.message
    (nb,) = verify_calibration(calib, {0: Frame(0, np.full_like(rgb, 120), 1, 0, 0)}, 2.0)
    assert nb.state == "no_board" and "not verified" in nb.message
    (nf,) = verify_calibration(calib, {0: None}, 2.0)
    assert nf.state == "no_frame"


def test_live_frames_are_undistorted_with_nonzero_distortion():
    calib, cal, rgb, T = _board_setup()
    src = SyntheticCameraSource(0, rgb, 0, fps=100)  # raw lens image, calib_version 0
    src.start()
    try:
        pipe = LivePipeline({0: src}, calib, make_depth_factory("oracle"))
        f = pipe.frame_for(0)
        assert f.calib_version == 7 and f.rgb.shape == rgb.shape
        assert pipe.frame_for(0).rgb is f.rgb or pipe.frame_for(0).seq >= f.seq  # cached per seq
        undist_rms = _rms(cal, SPEC, np.ascontiguousarray(f.rgb[..., ::-1]), T)
        raw_rms = _rms(cal, SPEC, np.ascontiguousarray(rgb[..., ::-1]), T)  # same pinhole model, no undistortion
    finally:
        src.stop()
    assert undist_rms < 1.0 and raw_rms > 3.0, (undist_rms, raw_rms)  # the lens model matters; live path applies it


# ---------------------------------------------------------------------------------------------- stored calibration
def _store_synthetic_calibration(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cams = [{"cam_id": c, "width": CAMS[c].width, "height": CAMS[c].height, "K_raw": CAMS[c].K.tolist(),
             "dist": [0.0] * 5, "K": CAMS[c].K.tolist(), "alpha": 0.0,
             "T_world_from_camera": CAMS[c].T_world_from_camera.tolist()} for c in sorted(CAMS)]
    (root / "calib_v001.yaml").write_text(yaml.safe_dump({"calib_version": CAMS[0].calib_version, "created": "synthetic",
                                                         "board": SPEC.to_dict(), "cameras": cams}))
    (root / "active").write_text("1\n")


class _StatusSource:
    """# MOCK: wraps a real source; ``override`` forces a status (e.g. ``failed``) without waiting for timeouts."""

    def __init__(self, inner):
        self.inner, self.override = inner, None
        self.cam_id = inner.cam_id

    def latest(self):
        return self.inner.latest()

    @property
    def status(self):
        return self.override or self.inner.status


def _clip(path: Path, cam: int, mutate=None, n=24):
    rng = np.random.default_rng(10 + cam)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    base = TRUTH[cam].rgb if mutate is None else mutate(TRUTH[cam].rgb)
    for _ in range(n):
        wr.write(cv2.cvtColor(_noisy(base, rng, 2.0), cv2.COLOR_RGB2BGR))
    wr.release()
    return path


def _file(path, cam):
    s = VideoFileSource(cam, path, realtime=False, loop=True)  # frames as fast as decoded; raw (calib_version 0)
    s.start()
    return s


class _Factory:
    """Counts depth-backend factory calls and keeps weak references to the oracle backends it built."""

    def __init__(self):
        self.inner, self.calls, self.refs = make_depth_factory("oracle"), 0, []

    def __call__(self, cals):
        self.calls += 1
        out = self.inner(cals)
        self.refs += [weakref.ref(b.backend) for b in out.values()]
        return out


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("live")
    root = d / "calib"
    _store_synthetic_calibration(root)
    calibration = load_calibration(root)  # the stored calibration, loaded as the CLI does
    paths = {c: _clip(d / f"cam{c}.mp4", c) for c in CAMS}
    srcs = {c: _StatusSource(_file(paths[c], c)) for c in CAMS}
    acq = load_acquisition_config()
    acq.update(seconds=0.3, max_frames=3, min_frames=3)
    fac = _Factory()
    pipe = LivePipeline(srcs, calibration, fac, live_cfg=replace(LiveConfig.from_yaml(), change=FAST),
                        acq_cfg=acq, grid_cfg=GRID, out_dir=d / "run", max_w=2560, max_h=720)
    pipe.startup()
    yield pipe, srcs, fac, d, paths
    for s in srcs.values():
        s.inner.stop()
    pipe.close()


def _step(pipe, n=1, dt=0.002):  # tiny dt: the sim barely advances, the monitor/compositor path is what is tested
    lf = None
    for _ in range(n):
        lf = pipe.step(dt)
        pipe.present(lf)
    return lf


def test_startup_reaches_simulating_and_releases_models(rig):
    pipe, _, fac, _, _ = rig
    assert pipe.life.history == [State.UNINITIALIZED, State.CALIBRATING, State.ACQUIRING, State.READY, State.SIMULATING]
    assert pipe.life.state is State.SIMULATING and fac.calls == 1
    assert [v.state for v in pipe.verify] == ["no_board", "no_board"]  # clips show the table, not the board: warn only
    gc.collect()
    assert all(r() is None for r in fac.refs)  # perception models released after READY (P1R6)
    assert len(pipe.ctl.snap.objects) == 4 and pipe.monitor.has_reference
    assert pipe.report["plane"]["agrees"]


def test_click_ignites_and_composites_live_frames(rig):
    pipe, _, _, d, _ = rig
    lf0 = _step(pipe, 2)
    assert lf0.image.shape[0] >= 720 and lf0.image.shape[1] == 2 * 1280 and lf0.cams == [0, 1]
    snap = pipe.ctl.snap
    obj = max((o.obj_id for o in snap.objects if o.obj_id), key=lambda o: (snap.obj_id == o).sum())
    uv, idx = project_elements(snap, 0)
    pick = None
    for (u, v), i in zip(uv[::5], idx[::5]):
        if snap.obj_id[i] == obj and snap.normal[i, 2] > 0.9:
            x, y = pipe._layout.from_view(0, float(u), float(v))
            r = pipe.click_display(round(x), round(y), "left")  # display pixel -> view pixel -> controller
            if r is not None and r.reason == "ignited":
                pick = r
                break
    assert pick is not None and pick.n_lit > 0 and pick.event.cam == 0 and pick.event.u is not None
    assert pipe.ctl.events[-1].kind == "ignite"
    lf = _step(pipe, 12, dt=1 / 30)
    st = pipe.ctl.sim.state_host()
    assert int(st.burning.sum()) > 0 and float(st.T_s.max()) > 600.0  # elements are actually burning, hot
    assert pipe.ctl.sim.t_sim > 0.2
    out = d / "composited.png"  # composited frames written
    cv2.imwrite(str(out), lf.image[..., ::-1])
    assert out.stat().st_size > 10_000
    assert np.abs(lf.image.astype(int) - lf0.image.astype(int)).max() > 20  # fire/char changed the picture
    assert any(ln.startswith("selected") for ln in pipe.ctl.hud_lines())


def test_overlay_toggle_draws_elements(rig):
    pipe, *_ = rig
    base = _step(pipe, 1).image.copy()
    pipe.key("o")
    on = _step(pipe, 1).image
    pipe.key("o")
    assert pipe.ctl.overlay is False
    assert np.abs(on.astype(int) - base.astype(int)).sum(axis=2).astype(bool).mean() > 0.002


def test_latency_csv_columns_and_values(rig):
    pipe, _, _, d, _ = rig
    _step(pipe, 5)
    pipe.recorder._fh.flush()
    with open(d / "run" / "latency.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == LATENCY_COLUMNS
    assert len(rows) >= 10 and {r["cam"] for r in rows} == {"0", "1"}
    for r in rows:
        age, c2d = float(r["frame_age_ms"]), float(r["capture_to_display_ms"])
        assert 0 <= age <= c2d < 60_000  # t_present - t_arrive includes the age at use
        assert int(r["t_present_ns"]) >= int(r["t_use_ns"]) >= int(r["t_arrive_ns"])
    s = pipe.recorder.stats()
    assert s["c2d_p95_ms"] >= s["c2d_p50_ms"] > 0 and s["fps"] > 0
    assert any("latency capture->display" in ln for ln in pipe._hud_extra(set()))


def test_static_noisy_clip_does_not_pause(rig):
    pipe, srcs, *_ = rig
    seq0 = srcs[0].latest().seq
    for _ in range(40):
        _step(pipe, 1)
        assert pipe.life.state is State.SIMULATING and not pipe.scene_changed
    assert srcs[0].latest().seq > seq0 + 5  # the monitor really saw new frames
    assert max(v for v in pipe.monitor.last.fraction.values()) < 0.01


def test_camera_drop_single_view_and_recovery(rig):
    pipe, srcs, *_ = rig
    t = [100.0]
    pipe.health.clock = lambda: t[0]
    srcs[1].override = "failed"
    _step(pipe)
    assert pipe.pane_cams == [0, 1]  # failed, but not for drop_seconds yet
    t[0] += pipe.cfg.drop_seconds + 0.1
    lf = _step(pipe)
    assert pipe.pane_cams == [0] and lf.cams == [0] and lf.image.shape[1] == 1280
    assert pipe.life.degraded and pipe.life.label == "SIMULATING+DEGRADED"
    assert any("dropped" in ln for ln in pipe._hud_extra({1}))
    assert pipe.click_display(100, 100) is not None  # single view still clickable
    assert pipe.click_display(1280 + 100, 100) is None  # no second pane any more
    srcs[1].override = None
    lf = _step(pipe)
    assert pipe.pane_cams == [0, 1] and not pipe.life.degraded and lf.image.shape[1] == 2 * 1280
    assert [e[2] for e in pipe.health.events] == ["dropped", "restored"]


def test_scene_change_pauses_then_reacquire(rig):
    pipe, srcs, fac, d, _paths = rig
    old = srcs[0].inner
    changed = _file(_clip(d / "cam0_changed.mp4", 0, _moved), 0)
    srcs[0].inner = changed  # the real scene changed: an object appeared in camera 0
    for _ in range(80):
        _step(pipe)
        if pipe.life.state is State.PAUSED:
            break
    assert pipe.life.state is State.PAUSED and pipe.scene_changed
    assert "scene changed: press r to re-acquire" in pipe.ctl.message
    assert any("SCENE CHANGED" in ln for ln in pipe._hud_extra(set()))
    t_sim = pipe.ctl.sim.t_sim
    _step(pipe, 3)
    assert pipe.ctl.sim.t_sim == t_sim  # paused: world not advanced, not rewritten
    snap_before = pipe.ctl.snap
    # space = continue on the old snapshot: monitor re-baselines instead of re-firing at once
    pipe.key("space")
    _step(pipe, 12)
    assert pipe.life.state is State.SIMULATING and not pipe.scene_changed and pipe.ctl.snap is snap_before
    # change it again (back to the original clip) and re-acquire this time
    srcs[0].inner = old
    for _ in range(80):
        _step(pipe)
        if pipe.life.state is State.PAUSED:
            break
    assert pipe.life.state is State.PAUSED
    pipe.key("r")
    assert pipe.life.state is State.ACQUIRING and pipe.ctl.reacquire_requested
    msgs = []
    pipe.progress = msgs.append
    _step(pipe)  # performs the blocking re-acquisition, then composites on the new snapshot
    assert pipe.life.state is State.SIMULATING and not pipe.scene_changed
    assert pipe.ctl.snap is not snap_before and pipe.snapshot_version == 2 and fac.calls == 2
    assert any("depth" in m for m in msgs) and any("capturing" in m for m in msgs)
    assert State.ACQUIRING in pipe.life.history[-4:]
    _step(pipe, 15)
    assert pipe.life.state is State.SIMULATING  # reference was re-captured: no immediate re-trigger
    changed.stop()
    pipe.close()
    ev = (d / "run" / "events.jsonl").read_text().strip().splitlines()
    assert len(ev) >= 1 and (d / "run" / "summary.json").is_file()
