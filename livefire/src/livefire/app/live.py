"""Live mode end to end (P1-006): calibration -> acquisition -> READY -> simulation on live frames, click-to-ignite.

``LivePipeline`` is the pure, headless-testable core: it owns the lifecycle (PRD 5.1), runs acquisition (P1-005) on
the started camera sources, releases the perception models at READY (P1R6), builds simulation + compositor + the P1-002
``InteractiveController`` on the frozen snapshot, and for every display frame composites the *live undistorted* frames
with the fire (``step``), updates the change monitor, camera health and latency records, and returns the display image.
``run_live`` is the thin OpenCV window loop around it.

Frames and units: world metres, +Z up, table plane z = 0; camera OpenCV; frames are RGB uint8 undistorted to the stored
calibration; times are ``time.perf_counter_ns`` (PRD 5.5) or seconds from an injectable clock. Host side only apart from
the Warp simulation/compositor. Single-threaded: call from one (the main) thread; sources run their own threads.

Latency (PRD 5.5), one CSV row per camera per presented frame (``latency.csv`` in the run directory):
``capture_to_display_ms = (t_present - t_arrive)`` where ``t_arrive`` is the source's arrival stamp of the frame used and
``t_present`` is stamped by the caller (``present``) right after the frame was shown (after ``imshow`` + ``waitKey``);
``frame_age_ms = t_use - t_arrive`` with ``t_use`` stamped just before that camera is composited; ``sim_lag_s`` =
wall time spent SIMULATING minus simulated time (time spent PAUSED does not count); ``fps`` = presented frames per
second over the HUD window. Glass-to-glass offset is measured separately (``latency-timer``, README).
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.app.build import build_sim
from livefire.app.change_monitor import ChangeConfig, ChangeMonitor, footprint_mask
from livefire.app.hud import HudInfo, draw_hud, make_ladder
from livefire.app.interactive import ClickResult, InteractiveController, ViewLayout
from livefire.app.lifecycle import ALLOWED, Lifecycle, State
from livefire.calib.board import BoardSpec, CalibrationError, board_corners_world, detect_charuco
from livefire.calib.store import CalibrationSet
from livefire.capture.types import Frame
from livefire.common.config import load_params, load_yaml
from livefire.common.errors import ReconstructionError
from livefire.common.geometry import CameraCalibration, invert_transform
from livefire.common.log import get_logger
from livefire.fusion.objects import load_acquisition_config
from livefire.render.compositor import Compositor
from livefire.render.geometry_overlay import draw_overlay
from livefire.scene.acquire import acquire_with_report, prepare_frame
from livefire.sim.materials import MaterialTable

log = get_logger("livefire.app.live")
WINDOW = "LiveFire live"
LATENCY_COLUMNS = (
    "frame", "cam", "seq", "state", "t_arrive_ns", "t_use_ns", "t_present_ns", "frame_age_ms",
    "capture_to_display_ms", "composite_ms", "frame_ms", "dropped_frame", "sim_lag_s", "fps", "cams_shown",
)  # fmt: skip
MIN_BOARD_CORNERS = 6


def _val(x: Any) -> Any:
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass(frozen=True)
class LiveConfig:
    """``configs/live.yaml`` (all values untuned defaults)."""

    verify_px: float = 2.0
    drop_seconds: float = 3.0
    change: ChangeConfig = field(default_factory=ChangeConfig)
    window_frames: int = 120
    max_wall_dt_s: float = 0.25  # one frame never advances the simulation by more than this much wall time

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any] | None = None) -> LiveConfig:
        c = cfg if cfg is not None else load_yaml("live")
        return cls(float(_val(c["verify_px"])), float(_val(c["drop_seconds"])), ChangeConfig.from_yaml(c["change"]),
                   int(c["latency"]["window_frames"]), float(_val(c["max_wall_dt_s"])))  # fmt: skip


# ---- calibration verification ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class CamVerify:
    """CALIBRATING check of one camera: ``state`` ``ok | high | no_board | no_frame``; ``rms_px`` board reprojection
    (NaN unless the board was seen); ``message`` for the HUD/log. ``high`` and ``no_board`` only warn."""

    cam: int
    state: str
    rms_px: float
    n_corners: int
    message: str


def verify_calibration(calibration: CalibrationSet, frames: Mapping[int, Frame | None], verify_px: float,
                       spec: BoardSpec | None = None) -> list[CamVerify]:
    """If the ChArUco board is visible in a camera's *raw* frame, project its corners through the stored pose and
    ``K_raw``/``dist`` and compare with the detected corners; warn if the RMS exceeds ``verify_px`` (the camera or table
    may have moved, or the board is not where the calibration put it). No board in view: warn that it is unverified.
    Raises ``CalibrationError`` if a frame's resolution differs from the calibration."""
    spec = spec or calibration.board
    out: list[CamVerify] = []
    for c, rec in sorted(calibration.cameras.items()):
        cal, f = rec.calibration, frames.get(c)
        if f is None:
            out.append(CamVerify(c, "no_frame", float("nan"), 0, f"cam{c}: no frame, calibration not verified"))
            continue
        if f.rgb.shape[:2] != (cal.height, cal.width):
            raise CalibrationError(f"cam{c}: calibration is {cal.width}x{cal.height}, camera delivers "
                                   f"{f.rgb.shape[1]}x{f.rgb.shape[0]}; recalibrate or change the capture resolution")
        raw_bgr = np.ascontiguousarray(f.rgb[..., ::-1])
        corners, ids = detect_charuco(spec, raw_bgr)
        if len(ids) < MIN_BOARD_CORNERS:
            out.append(CamVerify(c, "no_board", float("nan"), len(ids),
                                 f"cam{c}: board not visible, calibration not verified"))
            continue
        already_undistorted = f.calib_version == cal.calib_version
        K, dist = (cal.K, np.zeros(5)) if already_undistorted else (rec.K_raw, cal.dist)
        T_cw = invert_transform(cal.T_world_from_camera)
        rvec = cv2.Rodrigues(T_cw[:3, :3])[0]
        proj, _ = cv2.projectPoints(board_corners_world(spec)[ids], rvec, T_cw[:3, 3], K, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
        if rms > verify_px:
            out.append(CamVerify(c, "high", rms, len(ids), f"cam{c}: board reprojection {rms:.2f} px > {verify_px:.1f} px"
                                 " (camera or table moved? consider recalibrating)"))  # fmt: skip
        else:
            out.append(CamVerify(c, "ok", rms, len(ids), f"cam{c}: calibration verified, board reprojection {rms:.2f} px"))
    return out


# ---- camera health ---------------------------------------------------------------------------------------------
class CameraHealth:
    """Tracks which cameras are removed from rendering: a camera whose source status has been ``failed`` for
    ``drop_seconds`` (by ``clock``) is dropped and restored as soon as its status is anything else. Pure; ``events``
    lists ``(t, cam, "dropped" | "restored")``."""

    def __init__(self, drop_seconds: float, clock: Callable[[], float] = time.perf_counter) -> None:
        self.drop_seconds, self.clock = drop_seconds, clock
        self._since: dict[int, float] = {}
        self.dropped: set[int] = set()
        self.events: list[tuple[float, int, str]] = []

    def update(self, statuses: Mapping[int, str]) -> set[int]:
        now = self.clock()
        for c, st in statuses.items():
            if st == "failed":
                self._since.setdefault(c, now)
                if c not in self.dropped and now - self._since[c] >= self.drop_seconds:
                    self.dropped.add(c)
                    self.events.append((now, c, "dropped"))
            else:
                self._since.pop(c, None)
                if c in self.dropped:
                    self.dropped.discard(c)
                    self.events.append((now, c, "restored"))
        return set(self.dropped)


# ---- latency ---------------------------------------------------------------------------------------------------
class LatencyRecorder:
    """Per-frame latency rows (``LATENCY_COLUMNS``) to a CSV (if ``path``) and rolling statistics over the last
    ``window`` presented frames. ``period_s``: target frame period; a frame whose interval exceeds ``1.5 x period_s``
    counts as dropped. Rows are flushed every 30 rows and on ``close``."""

    def __init__(self, path: Path | None, window: int = 120, period_s: float = 1 / 30) -> None:
        self.path, self.period_s = path, period_s
        self.c2d: deque[float] = deque(maxlen=window * 2)
        self.age: deque[float] = deque(maxlen=window * 2)
        self.frame_ms: deque[float] = deque(maxlen=window)
        self.present_s: deque[float] = deque(maxlen=window)
        self.n_frames = self.n_dropped = 0
        self._fh = None
        self._w = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w", newline="")  # noqa: SIM115 - closed in close()
            self._w = csv.writer(self._fh)
            self._w.writerow(LATENCY_COLUMNS)

    def fps(self) -> float:
        p = self.present_s
        return (len(p) - 1) / (p[-1] - p[0]) if len(p) > 1 and p[-1] > p[0] else 0.0

    def add_frame(self, t_present_ns: int, frame_ms: float) -> bool:
        """Register one presented frame; returns whether it counts as dropped."""
        self.present_s.append(t_present_ns / 1e9)
        dropped = self.n_frames > 0 and frame_ms > 1.5e3 * self.period_s
        if self.n_frames > 0:
            self.frame_ms.append(frame_ms)
        self.n_frames += 1
        self.n_dropped += int(dropped)
        return dropped

    def add_row(self, row: dict[str, Any]) -> None:
        self.c2d.append(float(row["capture_to_display_ms"]))
        self.age.append(float(row["frame_age_ms"]))
        if self._w is not None:
            self._w.writerow([row[k] for k in LATENCY_COLUMNS])
            if self.n_frames % 30 == 0 and self._fh is not None:
                self._fh.flush()

    @staticmethod
    def _pct(d: deque[float], q: float) -> float:
        return float(np.percentile(np.fromiter(d, float), q)) if d else 0.0

    def stats(self) -> dict[str, float]:
        """p50/p95/p99 capture-to-display [ms], median frame age [ms], p95/p99 frame time [ms], delivered fps,
        dropped-frame rate."""
        return {
            "c2d_p50_ms": self._pct(self.c2d, 50), "c2d_p95_ms": self._pct(self.c2d, 95),
            "c2d_p99_ms": self._pct(self.c2d, 99), "age_p50_ms": self._pct(self.age, 50),
            "frame_p95_ms": self._pct(self.frame_ms, 95), "frame_p99_ms": self._pct(self.frame_ms, 99),
            "fps": self.fps(), "dropped_rate": self.n_dropped / max(1, self.n_frames - 1),
        }  # fmt: skip

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = self._w = None


# ---- depth backends --------------------------------------------------------------------------------------------
def make_depth_factory(name: str | None = None, device: str | None = None) -> Callable[[dict[int, CameraCalibration]],
                                                                                    dict[int, Any]]:
    """Factory ``{cam_id: CameraCalibration} -> {cam_id: AnchoredDepthBackend}``, called once per acquisition so the
    models can be freed at READY (P1R6). Anchors are per camera (P1R3); one network is shared by all cameras.
    ``name``: a ``configs/depth.yaml`` backend, or ``oracle`` (# MOCK: synthetic ground truth, tests/demo only)."""
    from livefire.perception.depth import AnchoredDepthBackend, OracleDepthBackend, TableRegion

    name = name or load_yaml("depth")["backend"]

    def factory(cals: dict[int, CameraCalibration]) -> dict[int, Any]:
        region = TableRegion.from_config()
        if name == "oracle":
            return {c: AnchoredDepthBackend(OracleDepthBackend(cal), cal, region) for c, cal in cals.items()}
        from livefire.perception.depth.hf_backends import create_depth_backend

        base = create_depth_backend(name, device=device)
        return {c: AnchoredDepthBackend(base, cal, region) for c, cal in cals.items()}

    return factory


# ---- the pipeline ----------------------------------------------------------------------------------------------
@dataclass
class LiveFrame:
    """One composed display frame: ``image`` RGB uint8 (H, W, 3), the camera ids in pane order and, per camera, the
    arrival stamp / seq of the frame used, ``t_use_ns`` and ``composite_ms``. Pass to ``LivePipeline.present``."""

    image: NDArray[np.uint8]
    cams: list[int]
    t_arrive_ns: dict[int, int]
    seq: dict[int, int]
    t_use_ns: dict[int, int]
    composite_ms: dict[int, float]
    state: str
    reacquired: bool = False  # this step performed a blocking re-acquisition (the window resets its wall clock)


class LivePipeline:
    """See module docstring. ``sources``: started ``CameraSource`` objects ``{cam_id: source}`` delivering raw (distorted)
    frames (``calib_version`` 0); ``calibration``: ``CalibrationSet``; ``depth_factory``: see ``make_depth_factory``.
    The constructor is cheap (no model, no simulation); ``startup`` does CALIBRATING -> ACQUIRING -> READY ->
    SIMULATING. ``progress`` (optional ``str -> None``) is called from the calling thread during acquisition."""

    def __init__(self, sources: Mapping[int, Any], calibration: CalibrationSet,
                 depth_factory: Callable[[dict[int, CameraCalibration]], Mapping[int, Any]], *,
                 live_cfg: LiveConfig | None = None, acq_cfg: dict[str, Any] | None = None,
                 grid_cfg: dict[str, Any] | None = None, app_cfg: dict[str, Any] | None = None,
                 out_dir: Path | None = None, max_w: int = 1600, max_h: int = 900,
                 clock: Callable[[], float] = time.perf_counter, progress: Callable[[str], None] | None = None,
                 first_frame_timeout_s: float = 10.0,
                 scene_label: str = "LIVE CAMERAS (simulation on frozen snapshot)") -> None:
        self.sources = dict(sources)
        self.calibration = calibration
        self.cals = {c: r.calibration for c, r in calibration.cameras.items()}
        self.undist = {c: r.undistorter() for c, r in calibration.cameras.items()}
        missing = set(self.cals) - set(self.sources)
        if missing:
            raise ValueError(f"no camera source for calibrated cameras {sorted(missing)}")
        self.depth_factory = depth_factory
        self.cfg = live_cfg or LiveConfig.from_yaml()
        self.acq_cfg = acq_cfg or load_acquisition_config()
        self.grid_cfg = grid_cfg or load_yaml("grid")
        self.app = app_cfg or load_yaml("app")
        self.sim_params = load_params("sim")
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.max_w, self.max_h, self.progress, self.scene_label = max_w, max_h, progress, scene_label
        self.first_frame_timeout_s = first_frame_timeout_s
        self.life = Lifecycle()
        self.health = CameraHealth(self.cfg.drop_seconds, clock)
        self.monitor = ChangeMonitor(self.cfg.change)
        fps = float(self.app.get("fps", 30))
        self.recorder = LatencyRecorder(self.out_dir / "latency.csv" if self.out_dir else None,
                                        self.cfg.window_frames, 1.0 / fps)
        self.fps_target = fps
        self.ladder = make_ladder(float(self.app["degrade_volume_scale"]), float(self.app["degrade_step_factor"]))
        self.verify: list[CamVerify] = []
        self.report: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.scene_changed = False
        self.error: str | None = None
        self.ctl: InteractiveController | None = None
        self.comp: Compositor | None = None
        self.snapshot_version = 0
        self.event_log: list[Any] = []  # events of earlier snapshots (before a re-acquire)
        self._prep: dict[int, Frame] = {}
        self._sim_wall = 0.0
        self._rt = 0.0
        self._last_present_ns: int | None = None
        self._frame_no = 0
        self._pane_cams: list[int] = []
        self._layout: ViewLayout | None = None
        self._back_notified = False
        self._reacquired = False

    # ---- frames -------------------------------------------------------------------------------------------
    def frame_for(self, cam: int) -> Frame | None:
        """Newest frame of ``cam`` undistorted to the stored calibration and stamped with its version (cached per
        ``seq``: each frame is remapped once), or None before the first frame."""
        f = self.sources[cam].latest()
        if f is None:
            return None
        c = self._prep.get(cam)
        if c is not None and c.seq == f.seq and c.t_arrive_ns == f.t_arrive_ns:
            return c
        self._prep[cam] = prepare_frame(f, self.cals[cam], self.undist[cam])
        return self._prep[cam]

    # ---- start-up / re-acquisition ----------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if self.progress is not None:
            self.progress(msg)

    def startup(self) -> None:
        """UNINITIALIZED -> CALIBRATING (load + verify) -> ACQUIRING -> READY -> SIMULATING. On failure the lifecycle
        goes to ERROR and the exception propagates."""
        try:
            self.life.transition(State.CALIBRATING)
            self._say("waiting for the first camera frames")
            self._wait_first_frames(self.first_frame_timeout_s)
            self._say("calibration: verifying against the board (if visible)")
            raw = {c: self.sources[c].latest() for c in self.cals}
            self.verify = verify_calibration(self.calibration, raw, self.cfg.verify_px)
            for v in self.verify:
                (log.info if v.state == "ok" else log.warning)(v.message)
            self.life.transition(State.ACQUIRING)
            self._acquire_and_build()
        except Exception as e:
            self._fail(e)
            raise

    def _wait_first_frames(self, timeout_s: float = 10.0) -> None:
        """Wait (wall clock) until every calibrated camera has delivered a frame, at most ``timeout_s``."""
        t_end = time.perf_counter() + timeout_s
        while any(self.sources[c].latest() is None for c in self.cals) and time.perf_counter() < t_end:
            time.sleep(0.01)

    def _fail(self, e: BaseException) -> None:
        self.error = f"{type(e).__name__}: {e}"
        if State.ERROR in ALLOWED[self.life.state]:
            self.life.transition(State.ERROR)
        log.error("live mode error: %s", self.error)

    def _acquire_and_build(self) -> None:
        """Lifecycle is ACQUIRING: acquire, release perception (P1R6), READY, build the simulation and go SIMULATING."""
        self.snapshot_version += 1
        active = {c: r for c, r in self.calibration.cameras.items() if self.sources[c].latest() is not None}
        if not active:
            raise ReconstructionError("no calibrated camera delivers frames")
        if len(active) < len(self.cals):
            absent = sorted(set(self.cals) - set(active))
            log.warning("camera(s) %s deliver no frames: acquiring with camera(s) %s only (single view)", absent, sorted(active))
        calib = self.calibration if len(active) == len(self.cals) else replace(self.calibration, cameras=active)
        backends = self.depth_factory({c: r.calibration for c, r in active.items()})
        try:
            res = acquire_with_report(self.sources, calib, backends, self.acq_cfg, self.grid_cfg,
                                      version=self.snapshot_version, progress=self.progress)
        finally:
            del backends
            self._release_perception()
        self.report = res.report
        self.life.transition(State.READY)
        self._say("building simulation and renderer")
        if self.ctl is not None:
            self.event_log.extend(self.ctl.events)
        sim = build_sim(res.snapshot, MaterialTable.load(), self.sim_params, self.app)
        self.comp = Compositor(res.snapshot)
        self.ctl = InteractiveController(sim, res.snapshot, sim.materials, self.life, self.sim_params)
        self.ctl.start()
        self._sim_wall = 0.0
        self.scene_changed = False
        self._back_notified = False
        self.monitor.set_reference({c: f.rgb for c in res.snapshot.cameras if (f := self.frame_for(c)) is not None})
        n_rev = sum(1 for o in res.snapshot.objects if o.obj_id and o.material.requires_review)
        self.ctl.message = (f"ready: {len(res.snapshot.pos)} elements, {n_rev} objects default to "
                            f"'{self.acq_cfg['default_object_material']}' (right click + 1-8 to re-tag)")

    @staticmethod
    def _release_perception() -> None:
        """Free the perception models (P1R6): collect garbage and, if torch is loaded and CUDA is on, empty its cache."""
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reacquire(self) -> None:
        """Handle the user's ``r``: ACQUIRING -> new snapshot -> READY -> SIMULATING (fire state is reset; earlier
        events are kept in ``event_log``). Failure: ERROR with ``error`` set (the lifecycle table allows only that)."""
        assert self.ctl is not None
        try:
            self._acquire_and_build()
            self.ctl.message = "re-acquired: new snapshot, fire reset. " + self.ctl.message
        except Exception as e:  # noqa: BLE001 - any failure -> ERROR state with a message
            self._fail(e)

    # ---- interaction ---------------------------------------------------------------------------------------
    @property
    def pane_cams(self) -> list[int]:
        """Cameras shown in the last composed frame, left to right."""
        return list(self._pane_cams)

    def click_display(self, x: float, y: float, button: str = "left", mods: int = 0) -> ClickResult | None:
        """Mouse click at display pixel ``(x, y)`` of the last composed frame; None outside the panes."""
        if self.ctl is None or self._layout is None:
            return None
        hit = self._layout.to_view(x, y)
        if hit is None or hit[0] >= len(self._pane_cams):
            return None
        pane, u, v = hit
        return self.ctl.click(self._pane_cams[pane], u, v, button, mods)

    def key(self, k: str) -> str:
        assert self.ctl is not None
        return self.ctl.key(k)

    # ---- one display frame -----------------------------------------------------------------------------------
    def step(self, wall_dt: float) -> LiveFrame:
        """Advance the simulation by ``wall_dt`` s, composite the live frames of the shown cameras, run the change
        monitor and camera health, and return the display image (RGB, HUD included). Handles a pending re-acquire
        request first (blocking; progress goes to ``progress``)."""
        assert self.ctl is not None and self.comp is not None, "call startup() first"
        self._reacquired = False
        if self.ctl.reacquire_requested:
            self.reacquire()
            self._reacquired = True
            if self.life.state is State.ERROR:
                raise RuntimeError(self.error or "re-acquisition failed")
        ctl, comp = self.ctl, self.comp
        snap = ctl.snap
        in_snap = sorted(snap.cameras)
        wall_dt = min(wall_dt, self.cfg.max_wall_dt_s)  # never try to catch up on a long stall / blocking acquisition
        dropped = self.health.update({c: str(getattr(self.sources[c], "status", "connected")) for c in in_snap})
        missing = [c for c in self.cals if c not in snap.cameras]  # calibrated but absent from the snapshot
        if self.life.state in (State.SIMULATING, State.PAUSED):
            self.life.set_degraded(bool(dropped) or bool(missing))
        if missing and not self._back_notified and any(self.sources[c].latest() is not None for c in missing):
            self._back_notified = True
            ctl.message = f"camera(s) {missing} now deliver frames: press r to re-acquire with them"
        res = ctl.tick(wall_dt)
        if res is not None:
            self._rt = res.rt_factor
            self._sim_wall += wall_dt
        cams = [c for c in ctl.visible_cams() if c not in dropped]
        if not cams:  # the selected single view is a dropped camera: fall back to the remaining camera(s)
            cams = [c for c in in_snap if c not in dropped]
        c0 = snap.cameras[min(snap.cameras)]
        self._pane_cams = cams
        lay = ViewLayout(c0.width, c0.height, max(1, len(cams)), self.max_w, self.max_h)
        self._layout = lay
        panes: list[NDArray[np.uint8]] = []
        info: dict[str, dict[int, Any]] = {"arr": {}, "seq": {}, "use": {}, "ms": {}}
        monitor_in: dict[int, tuple[int, NDArray[np.uint8]]] = {}
        foot: dict[int, NDArray[np.bool_]] = {}
        for c in cams:
            f = self.frame_for(c)
            if f is None:
                panes.append(self._blank(lay, f"camera {c}: no frame yet"))
                continue
            t_use = time.perf_counter_ns()
            img = comp.render(c, f, ctl.sim)
            ms = (time.perf_counter_ns() - t_use) / 1e6
            info["arr"][c], info["seq"][c], info["use"][c], info["ms"][c] = f.t_arrive_ns, f.seq, t_use, ms
            monitor_in[c] = (f.seq, f.rgb)
            foot[c] = footprint_mask(f.rgb, img, self.cfg.change)
            panes.append(cv2.resize(img, (lay.pane_w, lay.pane_h), interpolation=cv2.INTER_AREA))
        if not panes:
            panes.append(self._blank(lay, "NO CAMERA: all cameras failed, waiting for reconnect"))
        for c in in_snap:  # the monitor watches every working camera, shown or not (no footprint where nothing was composited)
            if c not in monitor_in and c not in dropped and (f := self.frame_for(c)) is not None:
                monitor_in[c] = (f.seq, f.rgb)
        self._monitor(monitor_in, foot)
        shown = np.concatenate(panes, axis=1)
        for i, c in enumerate(cams):  # annotate the display image with pane-aware coordinates
            self._annotate(shown, c, i, lay)
        hud = HudInfo(self.life.label, self.recorder.fps(), self.recorder.stats()["frame_p95_ms"], self._rt,
                      ctl.sim.t_sim, self.ladder[0], {c: float(np.mean(m)) for c, m in snap.view_hole_mask.items()},
                      len(snap.pos), False, tuple(self._hud_extra(dropped, missing)), self.scene_label)  # fmt: skip
        return LiveFrame(draw_hud(shown, hud), cams, info["arr"], info["seq"], info["use"], info["ms"], self.life.label,
                         reacquired=self._reacquired)

    @staticmethod
    def _blank(lay: ViewLayout, text: str) -> NDArray[np.uint8]:
        img = np.zeros((lay.pane_h, lay.pane_w, 3), np.uint8)
        cv2.putText(img, text, (20, lay.pane_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 120), 2, cv2.LINE_AA)
        return img

    def _annotate(self, pane: NDArray[np.uint8], cam: int, idx: int, lay: ViewLayout) -> None:
        ctl = self.ctl
        assert ctl is not None
        if ctl.overlay:
            draw_overlay(pane, ctl.snap, cam, idx, lay, ctl.mat, ctl.selected)
        elif ctl.selected is not None:
            self._draw_selected(pane, cam, idx, lay)
        for ev in ctl.events:  # click markers
            if ev.kind == "ignite" and ev.cam == cam and ev.u is not None and ev.v is not None:
                x, y = lay.from_view(idx, ev.u, ev.v)
                cv2.drawMarker(pane, (round(x), round(y)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

    def _draw_selected(self, pane: NDArray[np.uint8], cam: int, idx: int, lay: ViewLayout) -> None:
        from livefire.render.geometry_overlay import SELECTED_COLOR, project_elements

        ctl = self.ctl
        assert ctl is not None
        uv, ii = project_elements(ctl.snap, cam)
        for (u, v), e in zip(uv, ii):
            if ctl.snap.obj_id[e] == ctl.selected:
                x, y = lay.from_view(idx, float(u), float(v))
                cv2.circle(pane, (round(x), round(y)), 2, SELECTED_COLOR, -1, cv2.LINE_AA)

    def _monitor(self, frames: dict[int, tuple[int, NDArray[np.uint8]]], foot: dict[int, NDArray[np.bool_]]) -> None:
        """Change monitor step (PRD 5.1). A trigger pauses the simulation (never rewrites the world). If the user
        resumes (space) after a trigger, the reference is re-baselined to the current frames and monitoring continues
        on the old snapshot; ``r`` re-acquires instead."""
        ctl = self.ctl
        assert ctl is not None
        if self.scene_changed and self.life.state is State.SIMULATING:  # user resumed after a trigger
            self.monitor.set_reference({c: rgb for c, (_, rgb) in frames.items()})
            self.scene_changed = False
            ctl.message = "continuing on the old snapshot (change reference re-baselined); r re-acquires"
            return
        if self.life.state not in (State.SIMULATING, State.PAUSED) or self.scene_changed:
            return
        st = self.monitor.update(frames, foot)
        if st.triggered:
            self.scene_changed = True
            if self.life.state is State.SIMULATING:
                self.life.transition(State.PAUSED)
            ctl.message = "scene changed: press r to re-acquire"
            log.warning("scene change detected (fractions %s)", {c: round(v, 3) for c, v in st.fraction.items()})

    def _hud_extra(self, dropped: set[int], missing: list[int] | None = None) -> list[str]:
        ctl = self.ctl
        assert ctl is not None
        s = self.recorder.stats()
        lines = list(ctl.hud_lines())
        lines.append(f"latency capture->display p50 {s['c2d_p50_ms']:.0f} / p95 {s['c2d_p95_ms']:.0f} ms   "
                     f"frame age {s['age_p50_ms']:.0f} ms   sim lag {self._sim_wall - ctl.sim.t_sim:+.2f} s   "
                     f"{s['fps']:.1f} fps")  # fmt: skip
        cam = "   ".join(f"cam{c}:{getattr(self.sources[c], 'status', '?')}" for c in sorted(self.cals))
        cal = "; ".join(v.message for v in self.verify if v.state != "ok") if self.verify else ""
        if dropped:
            cam += f"   DEGRADED: camera(s) {sorted(dropped)} dropped (single-view), re-added on reconnect"
        lines.append(cam + (f"   [{cal}]" if cal else ""))
        if missing:
            lines.append(f"DEGRADED: camera(s) {missing} delivered no frames at start-up (single view); r re-acquires with them once they work")
        if self.scene_changed:
            lines.append("SCENE CHANGED: press r to re-acquire (space continues on the old snapshot)")
        return lines

    # ---- presentation / records ------------------------------------------------------------------------------
    def present(self, lf: LiveFrame) -> None:
        """Call right after the frame was shown (``imshow`` + ``waitKey``, or after writing it): stamps ``t_present``
        and appends one latency row per camera shown."""
        assert self.ctl is not None
        t_p = time.perf_counter_ns()
        frame_ms = 0.0 if self._last_present_ns is None else (t_p - self._last_present_ns) / 1e6
        self._last_present_ns = t_p
        dropped = self.recorder.add_frame(t_p, frame_ms)
        lag = self._sim_wall - self.ctl.sim.t_sim
        for c in lf.cams:
            if c not in lf.t_arrive_ns:
                continue
            self.recorder.add_row({
                "frame": self._frame_no, "cam": c, "seq": lf.seq[c], "state": lf.state,
                "t_arrive_ns": lf.t_arrive_ns[c], "t_use_ns": lf.t_use_ns[c], "t_present_ns": t_p,
                "frame_age_ms": round((lf.t_use_ns[c] - lf.t_arrive_ns[c]) / 1e6, 3),
                "capture_to_display_ms": round((t_p - lf.t_arrive_ns[c]) / 1e6, 3),
                "composite_ms": round(lf.composite_ms[c], 3), "frame_ms": round(frame_ms, 3),
                "dropped_frame": int(dropped), "sim_lag_s": round(lag, 4), "fps": round(self.recorder.fps(), 3),
                "cams_shown": len(lf.cams),
            })  # fmt: skip
        self._frame_no += 1

    def save_screenshot(self, image: NDArray[np.uint8]) -> Path:
        """Write ``image`` (RGB) to the run directory; returns the path."""
        assert self.out_dir is not None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / f"screenshot_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(p), np.ascontiguousarray(image[..., ::-1]))
        return p

    def close(self) -> None:
        """Close the latency CSV and write ``events.jsonl`` (clicks and material overrides, all snapshots) and
        ``summary.json`` (lifecycle history, calibration check, acquisition report, latency statistics). Live sessions
        are not replayable as a P0 record: the snapshot comes from the cameras and is not stored."""
        self.recorder.close()
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        events = self.event_log + (self.ctl.events if self.ctl else [])
        (self.out_dir / "events.jsonl").write_text("\n".join(json.dumps(asdict(e)) for e in events) + ("\n" if events else ""))
        (self.out_dir / "summary.json").write_text(json.dumps({
            "lifecycle": [s.value for s in self.life.history], "error": self.error, "calibration_check":
            [asdict(v) for v in self.verify], "acquisition": self.report, "latency": self.recorder.stats(),
            "camera_events": self.health.events, "n_events": len(events), "t_sim": self.ctl.sim.t_sim if self.ctl else 0.0,
        }, indent=1, default=str))  # fmt: skip


# ---- window loop (thin) ----------------------------------------------------------------------------------------
def _splash(win: str, w: int, h: int, label: str, msg: str) -> None:
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 60), 2, cv2.LINE_AA)
    cv2.putText(img, msg, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.imshow(win, img)
    cv2.waitKey(1)


def run_live(cameras_path: str | None, calib: str, calib_root: Path, depth: str | None, grid: int | None, out: Path,
             max_w: int | None = None, max_h: int | None = None, frames: int | None = None) -> int:
    """``python -m livefire.app live``: open the cameras and the window and run until ``q``/Esc (or ``frames``
    frames). Returns 0, or 1 on a start-up/acquisition error (message printed, window closed)."""
    from livefire.app.window import fit_limits
    from livefire.calib.store import load_calibration
    from livefire.capture.factory import load_camera_configs, make_source

    calibration = load_calibration(calib_root, None if calib == "active" else int(calib))
    cfgs = [c for c in load_camera_configs(cameras_path) if int(c["cam_id"]) in calibration.cameras]
    missing = set(calibration.cameras) - {int(c["cam_id"]) for c in cfgs}
    if missing:
        raise SystemExit(f"calibration v{calibration.version} has cameras {sorted(missing)} that are not in the cameras file")
    max_w, max_h = fit_limits(max_w, max_h)
    sources: dict[int, Any] = {int(c["cam_id"]): make_source(c, calib_version=0) for c in cfgs}
    app_cfg = load_yaml("app")
    grid_cfg = {**load_yaml("grid"), **({"G": grid} if grid else {})}
    run_dir = out / time.strftime("live_%Y%m%d_%H%M%S")
    pipe = LivePipeline(sources, calibration, make_depth_factory(depth, str(app_cfg.get("device", "cpu"))),
                        grid_cfg=grid_cfg, app_cfg=app_cfg, out_dir=run_dir, max_w=max_w, max_h=max_h)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    pipe.progress = lambda m: _splash(WINDOW, 1280, 360, pipe.life.label, m)
    state: dict[str, Any] = {"click": None}

    def on_mouse(event: int, x: int, y: int, flags: int, _: Any) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            state["click"] = (x, y, "left" if event == cv2.EVENT_LBUTTONDOWN else "right", flags)

    cv2.setMouseCallback(WINDOW, on_mouse)
    rc, n = 0, 0
    try:
        for s in sources.values():
            s.start()
        pipe.startup()
        last = time.perf_counter()
        while not pipe.ctl.quit and (frames is None or n < frames):  # type: ignore[union-attr]
            now = time.perf_counter()
            lf = pipe.step(now - last)
            last = time.perf_counter() if lf.reacquired else now  # a blocking re-acquire is not simulated time
            if state["click"] is not None:  # mapped with the layout of the frame just composed
                x, y, button, mods = state["click"]
                state["click"] = None
                r = pipe.click_display(x, y, button, mods)
                if r is not None:
                    log.info("click: %s", r.message)
            cv2.imshow(WINDOW, np.ascontiguousarray(lf.image[..., ::-1]))
            code = cv2.waitKey(1)
            pipe.present(lf)
            if code != -1:
                code &= 0xFF
                pipe.key(chr(code) if code != 27 else "esc")
            if pipe.ctl.screenshot_requested:  # type: ignore[union-attr]
                pipe.ctl.screenshot_requested = False  # type: ignore[union-attr]
                pipe.ctl.message = f"saved {pipe.save_screenshot(lf.image)}"  # type: ignore[union-attr]
            n += 1
            spare = 1.0 / pipe.fps_target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
    except Exception as e:  # noqa: BLE001 - report and exit cleanly
        pipe.error = pipe.error or f"{type(e).__name__}: {e}"
        log.error("live mode stopped: %s", pipe.error)
        print(f"live mode stopped: {pipe.error}", file=sys.stderr)
        rc = 1
    finally:
        for s in sources.values():
            s.stop()
        cv2.destroyAllWindows()
        if pipe.ctl is not None or pipe.error:
            pipe.close()
            print(f"run recorded in {run_dir} (latency.csv, events.jsonl, summary.json)")
    return rc
