"""Pure interactive controller (P1-002): clicks and keys -> actions on the simulation, HUD messages, replay events.

No window, no OpenCV: the window layer (``livefire.app.window``) feeds ``click`` / ``key`` / ``tick`` events and
draws what the controller exposes. Every action that changes the simulation (ignite, material override) is applied
through ``livefire.app.replay.apply_event`` and appended to ``controller.events`` (the replay log format; clicks are
ignite events with extra ``cam, u, v`` fields), so a session replays to the same digest (CPU).

Frames and units: click pixels are full-resolution view pixels of the camera (u right, v down); world points are
metres in the table frame; ignition radius in metres; times in simulated seconds. Host side only; the controller owns
the event log and the per-element material view, and mutates the simulation it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from livefire.app.click import pick_element_at, pixel_to_world
from livefire.app.lifecycle import Lifecycle, State
from livefire.app.replay import Event, apply_event, config_snapshot, write_record
from livefire.common.config import Params
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.snapshot import SceneSnapshot
from livefire.sim.coupled import FireSimulation, FrameResult
from livefire.sim.materials import MaterialTable

RADIUS_STEP = 1.25  # multiplicative step of the +/- keys


@dataclass(frozen=True)
class ClickResult:
    """Outcome of a mouse click. ``reason``: ``ignited | hole | outside | no_depth | not flammable | no fuel |
    selected | no object``; ``point`` world (3,) m or None; ``event`` the recorded event (or None)."""

    reason: str
    message: str
    point: np.ndarray | None = None
    obj_id: int = -1
    n_lit: int = 0
    event: Event | None = None


@dataclass(frozen=True)
class ViewLayout:
    """Side-by-side pane layout of ``n_panes`` views of ``width`` x ``height`` px, scaled to fit
    ``max_w`` x ``max_h`` (never upscaled). Each pane is ``pane_w`` x ``pane_h`` display pixels; a display pixel
    ``(x, y)`` maps to the view pixel ``u = (x_in_pane + 0.5) / sx - 0.5`` (pixel-centre convention)."""

    width: int
    height: int
    n_panes: int
    max_w: int
    max_h: int

    @property
    def scale(self) -> float:
        return min(1.0, self.max_w / (self.width * self.n_panes), self.max_h / self.height)

    @property
    def pane_w(self) -> int:
        return max(1, round(self.width * self.scale))

    @property
    def pane_h(self) -> int:
        return max(1, round(self.height * self.scale))

    @property
    def sx(self) -> float:
        return self.pane_w / self.width

    @property
    def sy(self) -> float:
        return self.pane_h / self.height

    def to_view(self, x: float, y: float) -> tuple[int, float, float] | None:
        """Display pixel -> ``(pane index, u, v)`` in full-resolution view pixels, or None outside the panes."""
        if not (0 <= x < self.pane_w * self.n_panes and 0 <= y < self.pane_h):
            return None
        pane = int(x // self.pane_w)
        return pane, (x - pane * self.pane_w + 0.5) / self.sx - 0.5, (y + 0.5) / self.sy - 0.5

    def from_view(self, pane: int, u: float, v: float) -> tuple[float, float]:
        """Inverse of :meth:`to_view`: full-resolution view pixel -> display pixel (float)."""
        return pane * self.pane_w + (u + 0.5) * self.sx - 0.5, (v + 0.5) * self.sy - 0.5


class InteractiveController:
    """State machine for the interactive viewer. Feed it ``click`` / ``key`` / ``tick``; read ``message``,
    ``hud_lines()``, ``events`` and the flags (``quit``, ``overlay``, ``screenshot_requested``,
    ``reacquire_requested``). ``life`` must be READY (``start`` moves it to SIMULATING)."""

    def __init__(self, sim: FireSimulation, snapshot: SceneSnapshot, materials: MaterialTable, life: Lifecycle,
                 sim_params: Params) -> None:
        self.sim, self.snap, self.materials, self.life = sim, snapshot, materials, life
        meta = sim_params.meta("ignite_radius")
        self.r_min, self.r_max = (float(x) for x in meta["range"])
        self.radius = float(sim_params["ignite_radius"])
        self.mat = snapshot.material_id.copy()  # per-element material id, tracks overrides
        self.events: list[Event] = []
        self.selected: int | None = None
        self.message = "ready: left click ignites, right click selects"
        self.overlay = False
        self.view_mode = 0  # 0 = dual (all cameras), k >= 1 = only camera sorted(cameras)[k - 1]
        self.quit = False
        self.screenshot_requested = False
        self.reacquire_requested = False
        self.cam_ids = sorted(snapshot.cameras)

    # ---- lifecycle ----------------------------------------------------------------------------
    def start(self) -> None:
        """READY -> SIMULATING."""
        self.life.transition(State.SIMULATING)

    def tick(self, wall_dt: float) -> FrameResult | None:
        """Advance the simulation by ``wall_dt`` [s] of wall time (fixed dt inside); None while paused or
        re-acquiring."""
        if self.life.state is not State.SIMULATING:
            return None
        return self.sim.run_frame(wall_dt)

    def visible_cams(self) -> list[int]:
        """Camera ids shown in the window, left to right (all cameras in dual mode, else one)."""
        if self.view_mode == 0 or len(self.cam_ids) == 1:
            return list(self.cam_ids)
        return [self.cam_ids[self.view_mode - 1]]

    # ---- events -------------------------------------------------------------------------------
    def _record(self, ev: Event) -> Event:
        rec = apply_event(self.sim, ev, self.radius)
        self.events.append(rec)
        return rec

    def _obj_material(self, obj_id: int) -> str:
        return MATERIAL_CLASSES[int(np.argmax(self.snap.objects[obj_id].material.class_probabilities))]

    def click(self, cam: int, u: float, v: float, button: str = "left", mods: int = 0) -> ClickResult:
        """Mouse click at full-resolution pixel ``(u, v)`` of camera ``cam``. ``button`` is ``"left"`` (ignite,
        P1R5) or ``"right"`` (select the object). ``mods`` (modifier flags) is currently unused."""
        if button not in ("left", "right"):
            raise ValueError(f"unknown button {button!r}")
        p, reason = pixel_to_world(self.snap, cam, u, v)
        if p is None:
            self.message = {"hole": "hole: no surface at this pixel", "outside": "outside: click is outside the view",
                            "no_depth": "no_depth: no valid depth here"}[reason]
            return ClickResult(reason, self.message)
        elem, obj = pick_element_at(self.snap, p)
        if button == "right":
            if obj < 0 or obj not in self.snap.objects:
                self.message = "no object here"
                return ClickResult("no object", self.message, p)
            self.selected = obj
            self.message = f"selected object {obj} ({self.snap.objects[obj].name})"
            return ClickResult("selected", self.message, p, obj)
        if elem < 0:  # valid depth but no scene element near it: room wall / background, treated like a hole
            self.message = "hole: no scene surface at this pixel (room / background)"
            return ClickResult("hole", self.message, p)
        if not self.materials.flammable[int(self.mat[elem])]:
            name = self.materials.names[int(self.mat[elem])]
            self.message = f"not flammable ({name})"
            return ClickResult("not flammable", self.message, p, obj)
        rec = self._record(Event("ignite", self.sim.t_sim, point=tuple(float(x) for x in p), radius=self.radius,
                                 cam=int(cam), u=float(u), v=float(v)))  # fmt: skip
        n = int(rec.n_lit or 0)
        if n == 0:
            self.message = "no fuel: no flammable element within the ignition radius"
            return ClickResult("no fuel", self.message, p, obj, 0, rec)
        self.message = f"ignited {n} elements (r = {self.radius * 100:.1f} cm)"
        return ClickResult("ignited", self.message, p, obj, n, rec)

    def set_selected_material(self, material_index: int) -> str:
        """Override the selected object's material (index into MATERIAL_CLASSES); recorded as an override event."""
        if self.selected is None:
            self.message = "select an object first (right click)"
            return self.message
        name = MATERIAL_CLASSES[material_index]
        self._record(Event("override", self.sim.t_sim, obj_id=self.selected, material=name))
        self.snap.objects.override_material(self.selected, name)
        self.mat[self.snap.obj_id == self.selected] = material_index
        self.message = f"object {self.selected} -> {name}"
        return self.message

    def key(self, k: str) -> str:
        """Handle one key (single character, or ``"space"``, ``"esc"``). Returns the HUD message."""
        k = {" ": "space", "\x1b": "esc"}.get(k, k.lower() if len(k) == 1 else k)
        if k in "12345678" and len(k) == 1:
            return self.set_selected_material(int(k) - 1)
        if k == "space":
            if self.life.state is State.SIMULATING:
                self.life.transition(State.PAUSED)
                self.message = "paused"
            elif self.life.state is State.PAUSED:
                self.life.transition(State.SIMULATING)
                self.message = "resumed"
        elif k in ("+", "=", "-", "_"):
            f = RADIUS_STEP if k in ("+", "=") else 1.0 / RADIUS_STEP
            self.radius = float(min(self.r_max, max(self.r_min, self.radius * f)))
            self.message = f"ignition radius {self.radius * 100:.1f} cm"
        elif k == "o":
            self.overlay = not self.overlay
            self.message = f"geometry overlay {'on' if self.overlay else 'off'}"
        elif k == "v":
            if len(self.cam_ids) > 1:
                self.view_mode = (self.view_mode + 1) % (len(self.cam_ids) + 1)
            self.message = "view: " + ("dual" if len(self.visible_cams()) > 1 else f"camera {self.visible_cams()[0]}")
        elif k == "r":
            if self.life.state is State.SIMULATING:
                self.life.transition(State.PAUSED)
            if self.life.state is State.PAUSED:
                self.life.transition(State.ACQUIRING)
                self.reacquire_requested = True
                self.message = "re-acquire requested"
        elif k == "s":
            self.screenshot_requested = True
            self.message = "screenshot requested"
        elif k in ("q", "esc"):
            self.quit = True
        return self.message

    def finish_reacquire(self, message: str) -> None:
        """Called by the window layer once re-acquisition finished: ACQUIRING -> READY -> SIMULATING."""
        if self.life.state is State.ACQUIRING:
            self.life.transition(State.READY)
            self.life.transition(State.SIMULATING)
        self.reacquire_requested = False
        self.message = message

    # ---- HUD / record ---------------------------------------------------------------------------
    def hud_lines(self) -> list[str]:
        """Extra HUD lines: selected object, its material and review flag, ignition radius, last message."""
        if self.selected is None:
            sel = "selected: none (right click an object)"
        else:
            o = self.snap.objects[self.selected]
            sel = (f"selected: {self.selected} {o.name}   material {self._obj_material(self.selected)}   "
                   f"{'REQUIRES REVIEW' if o.material.requires_review else 'reviewed'}")  # fmt: skip
        return [sel, f"ignition radius {self.radius * 100:.1f} cm   {self.message}"]

    def save_session(self, out: Path, scene: str, grid_override: dict[str, Any] | None, app_cfg: dict[str, Any],
                     status: str = "ok") -> str:
        """Write the replayable record (events.jsonl, config_snapshot.yaml, run.json, final_state.npz) to ``out``
        and return the final digest."""
        out.mkdir(parents=True, exist_ok=True)
        final = self.sim.state_host()
        digest = final.digest()
        final.save_arrays(out / "final_state.npz")
        write_record(out, self.events, config_snapshot(grid_override), {
            "status": status, "error": None, "seed": int(app_cfg["seed"]), "scene": scene,
            "final_step": self.sim.step_count, "t_sim": self.sim.t_sim, "digest": digest,
            "device": app_cfg["device"], "mode": "interactive", "grid_G": int(self.snap.solid_mask.shape[0]),
            "lifecycle": self.life.state.value,
        })  # fmt: skip
        return digest
