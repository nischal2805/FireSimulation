"""HUD overlay and degradation ladder (PRD 5.6)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DegradationLevel:
    """One rung of the ladder. ``None`` means 'keep the configured render.yaml value'."""

    level: int
    name: str
    volume_scale: float | None
    step_factor: float | None


def make_ladder(volume_scale: float, step_factor: float) -> tuple[DegradationLevel, ...]:
    """P0 ladder: 0 nominal, 1 lower renderer volume_scale, 2 level 1 + larger ray-march step_factor.
    (Levels 3+ of PRD 5.6 are not implemented in P0.)"""
    return (
        DegradationLevel(0, "nominal", None, None),
        DegradationLevel(1, "lower volume_scale", volume_scale, None),
        DegradationLevel(2, "larger step_factor", volume_scale, step_factor),
    )


@dataclass
class HudInfo:
    """Values shown on the HUD. Times in ms/s, fractions in 0..1."""

    state: str
    fps: float
    p95_ms: float
    rt_factor: float
    t_sim: float
    degradation: DegradationLevel
    hole_frac: dict[int, float]
    n_elements: int
    offline: bool
    extra: tuple[str, ...] = ()  # additional lines (interactive viewer: selection, material, radius, message)


def hud_lines(h: HudInfo) -> list[str]:
    """The HUD text lines."""
    holes = " ".join(f"cam{c}={100 * f:.1f}%" for c, f in sorted(h.hole_frac.items()))
    deg = ("degradation ladder off" if h.offline
           else f"degradation L{h.degradation.level} ({h.degradation.name})")
    lines = [
        (f"{h.state}   fps {h.fps:5.2f}   p95 frame {h.p95_ms:7.1f} ms   sim rt {h.rt_factor:5.3f}x   "
         f"sim t {h.t_sim:6.3f} s"),
        f"{deg}   holes {holes}   elements {h.n_elements}",
        "SYNTHETIC SCENE" + ("   OFFLINE (not real-time)" if h.offline else ""),
    ]
    return lines + list(h.extra)


def draw_hud(img_rgb: NDArray[np.uint8], h: HudInfo) -> NDArray[np.uint8]:
    """Return a copy of ``img_rgb`` (H,W,3 uint8 sRGB) with a translucent HUD band at the top."""
    out = img_rgb.copy()
    lines = hud_lines(h)
    band = 16 + 26 * len(lines)
    out[:band] = (out[:band] * 0.35).astype(np.uint8)
    for i, ln in enumerate(lines):
        col = (255, 200, 60) if i == 2 else (235, 235, 235)
        cv2.putText(out, ln, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out
