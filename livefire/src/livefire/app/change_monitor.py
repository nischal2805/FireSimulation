"""Scene-change monitor (PRD 5.1): has the real scene moved since the reference captured at READY?

Pure NumPy/OpenCV, no window, no simulation. Frames are undistorted RGB uint8 (H, W, 3) of the *input* camera (never the
composite). Each frame is area-averaged to ``downscale_width`` px grayscale (0..1), which suppresses sensor noise. The
signed difference to the reference is taken on *static* pixels only (outside the fire/smoke footprint, see
:func:`footprint_mask`); its median (global gain/offset change from auto-exposure) is removed; a pixel counts as changed
when the remainder exceeds ``pixel_threshold``. If the changed fraction of any camera exceeds ``threshold`` for
``frames`` consecutive updates that brought a new frame, ``update`` returns ``triggered=True``. The monitor never touches
the world; the pipeline decides what to do (PAUSED + HUD message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.common.config import load_yaml


def _val(x: Any) -> Any:
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass(frozen=True)
class ChangeConfig:
    """Monitor tunables (``configs/live.yaml: change``)."""

    threshold: float = 0.03
    frames: int = 10
    pixel_threshold: float = 0.10
    downscale_width: int = 96
    footprint_threshold: float = 0.03
    footprint_dilate_px: int = 2
    min_static_fraction: float = 0.2

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any] | None = None) -> ChangeConfig:
        c = cfg if cfg is not None else load_yaml("live")["change"]
        return cls(float(_val(c["threshold"])), int(c["frames"]), float(_val(c["pixel_threshold"])),
                   int(c["downscale_width"]), float(_val(c["footprint_threshold"])), int(c["footprint_dilate_px"]),
                   float(c["min_static_fraction"]))  # fmt: skip


@dataclass
class ChangeStatus:
    """Result of one ``update``: ``fraction`` per camera (changed share of static pixels, NaN = no decision),
    ``consecutive`` (current run length), ``triggered`` (run reached ``frames``)."""

    fraction: dict[int, float] = field(default_factory=dict)
    consecutive: int = 0
    triggered: bool = False


def small_gray(rgb: NDArray[np.uint8], width: int) -> NDArray[np.float32]:
    """Area-averaged grayscale of an RGB uint8 frame at ``width`` px (aspect kept), float32 0..1."""
    h, w = rgb.shape[:2]
    size = (width, max(1, round(h * width / w)))
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def footprint_mask(frame_rgb: NDArray[np.uint8], composite_rgb: NDArray[np.uint8], cfg: ChangeConfig) -> NDArray[np.bool_]:
    """Fire/smoke footprint at monitor resolution: where the composite differs from the input frame (flame light,
    smoke, char, ash), dilated by ``footprint_dilate_px``. Inputs (H, W, 3) uint8 RGB of equal size."""
    a, b = small_gray(frame_rgb, cfg.downscale_width), small_gray(composite_rgb, cfg.downscale_width)
    m = (np.abs(a - b) > cfg.footprint_threshold).astype(np.uint8)
    if cfg.footprint_dilate_px > 0 and m.any():
        k = 2 * cfg.footprint_dilate_px + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m.astype(bool)


class ChangeMonitor:
    """See module docstring. Not thread-safe. ``set_reference`` must be called before ``update`` has any effect."""

    def __init__(self, cfg: ChangeConfig | None = None) -> None:
        self.cfg = cfg or ChangeConfig.from_yaml()
        self._ref: dict[int, NDArray[np.float32]] = {}
        self._last_seq: dict[int, int] = {}
        self.consecutive = 0
        self.last = ChangeStatus()

    @property
    def has_reference(self) -> bool:
        return bool(self._ref)

    def set_reference(self, frames: dict[int, NDArray[np.uint8]]) -> None:
        """Capture the reference appearance (``{cam_id: rgb}``) and reset the run counter. Cameras absent from
        ``frames`` keep their previous reference."""
        for c, rgb in frames.items():
            self._ref[c] = small_gray(rgb, self.cfg.downscale_width)
        self.consecutive = 0
        self._last_seq.clear()
        self.last = ChangeStatus()

    def reset_run(self) -> None:
        """Forget the current run of over-threshold updates (e.g. after the user acknowledged a trigger)."""
        self.consecutive = 0

    def changed_fraction(self, cam: int, rgb: NDArray[np.uint8], footprint: NDArray[np.bool_] | None = None) -> float:
        """Changed share of the static pixels of ``cam`` (NaN if there is no reference or too few static pixels)."""
        ref = self._ref.get(cam)
        if ref is None:
            return float("nan")
        cur = small_gray(rgb, self.cfg.downscale_width)
        if cur.shape != ref.shape:
            return float("nan")
        static = np.ones(cur.shape, bool) if footprint is None or footprint.shape != cur.shape else ~footprint
        if static.mean() < self.cfg.min_static_fraction:
            return float("nan")
        d = (cur - ref)[static]
        d = d - np.median(d)  # global exposure / gain change is not a scene change
        return float((np.abs(d) > self.cfg.pixel_threshold).mean())

    def update(self, frames: dict[int, tuple[int, NDArray[np.uint8]]],
               footprints: dict[int, NDArray[np.bool_]] | None = None) -> ChangeStatus:
        """One monitor step. ``frames``: ``{cam_id: (seq, rgb)}`` of the cameras currently rendered. The run counter
        advances only if some camera delivered a new ``seq`` since the previous update; it resets when no camera is
        over the threshold. The status keeps the previous fractions for cameras without a new frame."""
        fresh = [c for c, (seq, _) in frames.items() if self._last_seq.get(c) != seq]
        if not fresh or not self.has_reference:
            return self.last
        frac = dict(self.last.fraction)
        for c in fresh:
            seq, rgb = frames[c]
            self._last_seq[c] = seq
            frac[c] = self.changed_fraction(c, rgb, (footprints or {}).get(c))
        over = any(v > self.cfg.threshold for c, v in frac.items() if c in frames and np.isfinite(v))
        self.consecutive = self.consecutive + 1 if over else 0
        self.last = ChangeStatus(frac, self.consecutive, self.consecutive >= self.cfg.frames)
        return self.last
