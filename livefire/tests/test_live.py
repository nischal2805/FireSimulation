"""P1-006: live session on synthetic cameras (mock depth, identity-distortion calibration), headless."""

import numpy as np
import pytest

from livefire.app.live import LiveSession
from livefire.app.lifecycle import State
from livefire.calib.store import CalibrationSet, CameraRecord
from livefire.common.config import load_yaml
from livefire.fusion.objects import load_acquisition_config
from livefire.perception.depth import AnchoredDepthBackend, TableRegion
from livefire.perception.depth.base import RawDepth
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

CAMS = load_cameras()
TRUTH = {c: render_view(build_table_v1(), CAMS[c]) for c in CAMS}


class _Oracle:  # MOCK: ground-truth depth with a 3 % scale error that the table anchor must remove
    device = "cpu"
    peak_vram_allocated_mb = peak_vram_reserved_mb = 0.0

    def __init__(self, cam):
        self.t = TRUTH[cam]

    def warmup(self, shape):
        pass

    def infer_raw(self, frame):
        return RawDepth((1.03 * self.t.depth).astype(np.float32), self.t.valid.copy(), "metric", frame.calib_version)


def _factory(cals):
    reg = TableRegion.from_config()
    return {c: AnchoredDepthBackend(_Oracle(c), cal, reg) for c, cal in cals.items()}


@pytest.fixture()
def session(monkeypatch, tmp_path):
    cal = CalibrationSet(CAMS[0].calib_version, None, {c: CameraRecord(CAMS[c], CAMS[c].K, 0.0) for c in CAMS}, {})
    monkeypatch.setattr("livefire.app.live.load_calibration", lambda root: cal)
    cams_yaml = tmp_path / "cams.yaml"
    cams_yaml.write_text("cameras:\n  - {cam_id: 0, type: synthetic, source: 0, fps: 100}\n"
                         "  - {cam_id: 1, type: synthetic, source: 0, fps: 100}\n")
    acq = load_acquisition_config()
    acq.update(seconds=0.3, max_frames=6)
    monkeypatch.setattr("livefire.scene.acquire.load_acquisition_config", lambda: acq)
    s = LiveSession(str(cams_yaml), tmp_path, None, 32, 1600, 900, backends_factory=_factory)
    yield s
    s.close()


def test_live_acquire_click_ignite_compose(session):
    s = session
    assert s.life.state is State.READY and s.snapshot.calib_version == CAMS[0].calib_version
    s.controller.start()
    # find the paper object (default material is paper) and click a pixel on it in camera 0
    from livefire.app.click import pixel_to_world

    ids = [o.obj_id for o in s.snapshot.objects if o.obj_id]
    assert ids
    hit = None
    calib = s.snapshot.cameras[0]
    for v in range(0, calib.height, 6):
        for u in range(0, calib.width, 6):
            p, _ = pixel_to_world(s.snapshot, 0, u, v)
            if p is not None and s.snapshot.obj_id[np.argmin(np.linalg.norm(s.snapshot.pos - p, axis=1))] in ids:
                hit = (u, v)
                break
        if hit:
            break
    assert hit
    r = s.controller.click(0, hit[0], hit[1])
    assert r.reason == "ignited" and r.n_lit > 0, r.message
    for _ in range(5):
        img = s.compose(1 / 30)
    assert img.dtype == np.uint8 and img.shape[0] > 0


def test_reacquire(session):
    s = session
    s.controller.start()
    s.controller.key("r")
    assert s.controller.reacquire_requested
    s.reacquire()
    assert s.life.state is State.SIMULATING and not s.controller.reacquire_requested
