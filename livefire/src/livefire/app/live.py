"""Live mode (P1-006): real cameras -> calibration -> acquisition -> frozen snapshot -> fire over live frames.

``LiveSession`` is an ``InteractiveSession`` whose scene comes from live camera frames: load the active calibration,
start the configured cameras, acquire a snapshot (monocular depth anchored to the table plane, fused, clustered into
objects), release the depth model (P1R6), then simulate on the frozen snapshot while compositing every display
frame over the newest undistorted live frame. Click/keys are the P1-002 controller's.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from livefire.app.build import build_sim
from livefire.app.hud import make_ladder
from livefire.app.interactive import InteractiveController
from livefire.app.lifecycle import Lifecycle, State
from livefire.app.window import InteractiveSession, fit_limits, run_session
from livefire.calib.store import DEFAULT_ROOT, load_calibration
from livefire.capture.factory import load_camera_configs, make_source
from livefire.capture.types import Frame
from livefire.common.config import ConfigError, load_params, load_yaml
from livefire.common.log import get_logger
from livefire.render.compositor import Compositor
from livefire.scene.acquire import acquire_with_report
from livefire.sim.materials import MaterialTable

log = get_logger("livefire.app.live")


def _release_models(*objs: Any) -> None:
    """Drop perception models and free cached CUDA memory (P1R6)."""
    del objs
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class LiveSession(InteractiveSession):
    """Live cameras + real acquisition. Same window/controller API as ``InteractiveSession``."""

    def __init__(self, cameras_path: str | None = None, calib_root: Path = DEFAULT_ROOT, depth: str | None = None,
                 grid: int | None = None, max_w: int = 1600, max_h: int = 900, backends_factory: Any = None) -> None:
        self.scene, self.grid_override = "live", ({"G": grid} if grid else {})
        self.app = load_yaml("app")
        self.fps = float(self.app.get("fps", 30))
        self.max_w, self.max_h = max_w, max_h
        self.depth_name, self._backends_factory = depth, backends_factory
        self.life = Lifecycle()
        self._last: dict[int, Frame] = {}
        self.sources: list[Any] = []
        try:
            self.life.transition(State.CALIBRATING)
            self.cal = load_calibration(Path(calib_root))
            self.undist = {c: r.undistorter() for c, r in self.cal.cameras.items()}
            cfgs = [c for c in load_camera_configs(cameras_path) if int(c["cam_id"]) in self.cal.cameras]
            missing = set(self.cal.cameras) - {int(c["cam_id"]) for c in cfgs}
            if missing:
                raise ConfigError(f"calibrated cameras {sorted(missing)} are not listed in cameras.yaml")
            self.sources = [make_source(c, 0) for c in cfgs]  # raw frames: undistorted here, version stamped
            for s in self.sources:
                s.start()
            log.info("calibration v%d, cameras %s", self.cal.version, sorted(self.cal.cameras))
            self.life.transition(State.ACQUIRING)
            self._acquire_and_build()
            self.life.transition(State.READY)
        except Exception:
            self.life.transition(State.ERROR)
            self.close()
            raise
        self.ladder = make_ladder(float(self.app["degrade_volume_scale"]), float(self.app["degrade_step_factor"]))
        self._stamps = deque(maxlen=31)
        self.rt = 0.0

    def close(self) -> None:
        """Stop all camera threads."""
        for s in self.sources:
            s.stop()

    def _acquire_and_build(self) -> None:
        cals = {c: r.calibration for c, r in self.cal.cameras.items()}
        if self._backends_factory is not None:  # tests inject mock backends
            backends = self._backends_factory(cals)
        else:
            from livefire.perception.depth import AnchoredDepthBackend, TableRegion
            from livefire.perception.depth.hf_backends import create_depth_backend

            raw = create_depth_backend(self.depth_name)  # one network shared by all cameras, anchored per camera
            region = TableRegion.from_config()
            backends = {c: AnchoredDepthBackend(raw, cal, region) for c, cal in cals.items()}
        log.info("acquiring scene (keep the scene still)...")
        res = acquire_with_report(self.sources, self.cal, backends, grid_cfg={**load_yaml("grid"), **self.grid_override})
        del backends
        _release_models()
        self._finish_build(res)

    def _finish_build(self, res: Any) -> None:
        """Log the acquisition report and build the simulation, compositor and controller from ``res.snapshot``."""
        for w in res.report["warnings"]:
            log.warning("acquisition: %s", w)
        for o, d in res.report["objects"].items():
            log.info("object %s %s: %d elements, material %s%s", o, d["name"], d["elements"], d["material"],
                     " (review)" if d["requires_review"] else "")
        log.info("acquired in %.1f s", res.report["acquisition_s"])
        self.snapshot = res.snapshot
        self.report = res.report
        sim_params = load_params("sim")
        sim = build_sim(self.snapshot, MaterialTable.load(), sim_params, self.app)
        self.comp = Compositor(self.snapshot)
        self.controller = InteractiveController(sim, self.snapshot, sim.materials, self.life, sim_params)
        c0 = self.snapshot.cameras[min(self.snapshot.cameras)]
        self.size = (c0.width, c0.height)
        self.holes = {c: float(np.mean(m)) for c, m in self.snapshot.view_hole_mask.items()}

    def _frames(self) -> dict[int, Frame | None]:
        """Newest live frame per camera, undistorted and stamped with the calibration version. A camera that has
        not delivered a new frame keeps showing its last one."""
        ver = self.cal.version
        for s in self.sources:
            f = s.latest()
            if f is None or (s.cam_id in self._last and f.seq == self._last[s.cam_id].seq):
                continue
            self._last[s.cam_id] = Frame(s.cam_id, self.undist[s.cam_id].apply(f.rgb), f.t_arrive_ns, f.seq, ver)
        return {s.cam_id: self._last.get(s.cam_id) for s in self.sources}

    def reacquire(self) -> None:
        """Re-run acquisition (user pressed ``r``); the old simulation and material overrides are discarded."""
        try:
            self._acquire_and_build()
        except Exception as e:  # noqa: BLE001 - keep the old scene and tell the user
            log.error("re-acquire failed: %s", e)
            self.controller.finish_reacquire(f"re-acquire failed: {e}")
            return
        self.life.transition(State.READY)
        self.life.transition(State.SIMULATING)
        self.controller.message = "re-acquired: new scene frozen"
        self.controller.reacquire_requested = False


def run_live(cameras: str | None, calib_root: str | None, depth: str | None, grid: int | None, out: Path,
             max_w: int | None = None, max_h: int | None = None, frames: int | None = None) -> int:
    """``python -m livefire.app live``: build the live session, then run the window loop."""
    max_w, max_h = fit_limits(max_w, max_h)
    t0 = time.perf_counter()
    sess = LiveSession(cameras, Path(calib_root) if calib_root else DEFAULT_ROOT, depth, grid, max_w, max_h)
    log.info("ready after %.1f s: left click ignites, right click selects, 1-8 sets material, o overlay, r re-acquire, q quit",
             time.perf_counter() - t0)
    try:
        return run_session(sess, "live", out, frames, False)
    finally:
        sess.close()
