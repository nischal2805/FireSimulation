"""Thin OpenCV window layer for the interactive viewer (P1-002). All logic lives in ``interactive.py``/``click.py``.

``InteractiveSession`` builds the synthetic scene, simulation and renderer and turns controller state into a display
image (``compose``); ``run_interactive`` drives it in a window (mouse + keys). ``compose`` needs no display, so it
is also used headless (``--frames N`` smoke runs, tests)."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from livefire.app.build import acquire_snapshot, build_sim, load_synthetic_calibration
from livefire.app.hud import HudInfo, draw_hud, make_ladder
from livefire.app.interactive import InteractiveController, ViewLayout
from livefire.app.lifecycle import Lifecycle, State
from livefire.capture.synthetic_source import SyntheticCameraSource
from livefire.common.config import load_params, load_yaml
from livefire.common.geometry import project
from livefire.common.log import get_logger
from livefire.render.compositor import Compositor
from livefire.sim.materials import MaterialTable

log = get_logger("livefire.app.window")
WINDOW = "LiveFire interactive"
_PALETTE = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 255, 80), (255, 80, 255), (80, 255, 255)]


def screen_size() -> tuple[int, int] | None:
    """Primary screen size in pixels via tkinter, or None when unavailable (headless, no tkinter)."""
    try:
        import tkinter

        root = tkinter.Tk()
        try:
            root.withdraw()
            return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001 - any failure means "unknown"
        return None


def fit_limits(max_w: int | None, max_h: int | None) -> tuple[int, int]:
    """Window size limits: explicit values win; otherwise 95 % of the detected screen (minus a margin for the title
    bar and HUD), falling back to 1600x900."""
    scr = screen_size() if (max_w is None or max_h is None) else None
    dw, dh = (int(scr[0] * 0.95), int(scr[1] * 0.85)) if scr else (1600, 900)
    return (max_w if max_w is not None else dw), (max_h if max_h is not None else dh)


class InteractiveSession:
    """Synthetic-scene session: lifecycle, sources, simulation, renderer and the controller. Not thread-safe."""

    def __init__(self, scene: str = "table_v1", grid: int | None = None, max_w: int = 1600, max_h: int = 900) -> None:
        self.scene, self.grid_override = scene, ({"G": grid} if grid else {})
        self.app = load_yaml("app")
        self.life = Lifecycle()
        try:
            self.life.transition(State.CALIBRATING)  # synthetic calibration (cameras are not needed yet)
            scene_obj, cams = load_synthetic_calibration(scene)
            self.life.transition(State.ACQUIRING)
            self.snapshot, rgb = acquire_snapshot(scene_obj, cams, {**load_yaml("grid"), **self.grid_override})
            sim_params = load_params("sim")
            sim = build_sim(self.snapshot, MaterialTable.load(), sim_params, self.app)
            self.comp = Compositor(self.snapshot)
            fps = float(self.app.get("fps", 30))
            self.sources = [SyntheticCameraSource(c, rgb[c], cams[c].calib_version, fps) for c in sorted(rgb)]
            self.life.transition(State.READY)
        except Exception:
            self.life.transition(State.ERROR)
            raise
        self.fps = fps
        self.controller = InteractiveController(sim, self.snapshot, sim.materials, self.life, sim_params)
        c0 = self.snapshot.cameras[min(self.snapshot.cameras)]
        self.max_w, self.max_h, self.size = max_w, max_h, (c0.width, c0.height)
        self.ladder = make_ladder(float(self.app["degrade_volume_scale"]), float(self.app["degrade_step_factor"]))
        self.holes = {c: float(np.mean(m)) for c, m in self.snapshot.view_hole_mask.items()}
        self._stamps: deque[float] = deque(maxlen=31)
        self.rt = 0.0

    @property
    def layout(self) -> ViewLayout:
        return ViewLayout(self.size[0], self.size[1], len(self.controller.visible_cams()), self.max_w, self.max_h)

    def _overlay(self, img: np.ndarray, cam: int, pane: int, lay: ViewLayout) -> None:
        ctl = self.controller
        pos = self.snapshot.pos.astype(np.float64)
        calib = self.snapshot.cameras[cam]

        def dots(mask: np.ndarray, color: tuple[int, int, int], r: int) -> None:
            if not mask.any():
                return
            uv, _, ok = project(calib, pos[mask])
            for (u, v), o in zip(uv, ok):
                if o:
                    x, y = lay.from_view(pane, float(u), float(v))
                    cv2.circle(img, (round(x), round(y)), r, color, -1, cv2.LINE_AA)

        if ctl.overlay:
            ids = self.snapshot.obj_id
            for oid in np.unique(ids):
                dots(ids == oid, _PALETTE[int(oid) % len(_PALETTE)], 1)
        if ctl.selected is not None:
            dots(self.snapshot.obj_id == ctl.selected, (0, 255, 255), 2)
        for ev in ctl.events:  # click markers for this camera
            if ev.kind == "ignite" and ev.cam == cam and ev.u is not None and ev.v is not None:
                x, y = lay.from_view(pane, ev.u, ev.v)
                cv2.drawMarker(img, (round(x), round(y)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

    def _frames(self) -> dict[int, Any]:
        """Newest frame per camera (``Frame`` or None), stamped with the snapshot's calib_version."""
        return {s.cam_id: s.latest() for s in self.sources}

    def compose(self, wall_dt: float) -> np.ndarray:
        """Advance the sim by ``wall_dt`` s, render the visible views and return the display image (H,W,3 uint8
        RGB) with overlays and HUD."""
        ctl = self.controller
        res = ctl.tick(wall_dt)
        if res is not None:
            self.rt = res.rt_factor
        cams = ctl.visible_cams()
        lay = self.layout
        panes = []
        frames = self._frames()
        for i, c in enumerate(cams):
            f = frames[c]
            if f is None:
                raise RuntimeError("camera source produced no frame")
            img = self.comp.render(c, f, ctl.sim)
            img = cv2.resize(img, (lay.pane_w, lay.pane_h), interpolation=cv2.INTER_AREA)
            self._overlay(img, c, i, lay)
            panes.append(img)
        self._stamps.append(time.perf_counter())
        st = self._stamps
        fps = (len(st) - 1) / (st[-1] - st[0]) if len(st) > 1 and st[-1] > st[0] else 0.0
        info = HudInfo(self.life.label, fps, 0.0, self.rt, ctl.sim.t_sim, self.ladder[0], self.holes,
                       len(self.snapshot.pos), False, tuple(ctl.hud_lines()))  # fmt: skip
        return draw_hud(np.concatenate(panes, axis=1), info)

    def reacquire(self) -> None:
        """Re-acquire request: the synthetic scene is static, so the snapshot is unchanged."""
        self.controller.finish_reacquire("re-acquire: synthetic scene is static, snapshot unchanged")


def run_interactive(scene: str, grid: int | None, out: Path, max_w: int | None = None, max_h: int | None = None,
                    frames: int | None = None, record: bool = False) -> int:
    """Open the window and run until ``q``/Esc (or ``frames`` frames). Screenshots go to ``out``. With ``record``
    the session is also recorded to ``out`` (events, config, digest) so ``replay <out>`` reproduces it.
    ``max_w``/``max_h`` default to the detected screen size (fallback 1600x900)."""
    max_w, max_h = fit_limits(max_w, max_h)
    return run_session(InteractiveSession(scene, grid, max_w, max_h), scene, out, frames, record)


def run_session(sess: InteractiveSession, scene: str, out: Path, frames: int | None = None, record: bool = False) -> int:
    """Window loop for any session (synthetic or live). ``sess.controller`` may be replaced by a re-acquire."""
    ctl = sess.controller
    ctl.start()
    for s in sess.sources:
        s.start()
    state: dict[str, Any] = {"click": None}

    def on_mouse(event: int, x: int, y: int, flags: int, _: Any) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            state["click"] = (x, y, "left" if event == cv2.EVENT_LBUTTONDOWN else "right", flags)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    last, n, status = time.perf_counter(), 0, "ok"
    try:
        while not ctl.quit and (frames is None or n < frames):
            now = time.perf_counter()
            img = sess.compose(now - last)
            last = now
            if state["click"] is not None:  # mapped with the layout of the image just shown
                x, y, button, mods = state["click"]
                state["click"] = None
                hit = sess.layout.to_view(x, y)
                if hit is not None:
                    pane, u, v = hit
                    r = ctl.click(ctl.visible_cams()[pane], u, v, button, mods)
                    log.info("click: %s", r.message)
            cv2.imshow(WINDOW, np.ascontiguousarray(img[..., ::-1]))
            code = cv2.waitKey(1)
            if code != -1:
                code &= 0xFF
                ctl.key(chr(code) if code != 27 else "esc")
            if ctl.screenshot_requested:
                ctl.screenshot_requested = False
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(path), img[..., ::-1])
                ctl.message = f"saved {path}"
            if ctl.reacquire_requested:
                sess.reacquire()
                ctl = sess.controller
            n += 1
            spare = 1.0 / sess.fps - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
    except BaseException:
        status = "error"
        raise
    finally:
        for s in sess.sources:
            s.stop()
        cv2.destroyAllWindows()
        if record:
            digest = ctl.save_session(out, scene, sess.grid_override, sess.app, status)
            log.info("session recorded to %s (digest %s)", out, digest)
    return 0
