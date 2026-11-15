"""Geometry overlay (live mode ``o`` key): the reconstructed surface elements projected onto a live view, coloured by
material (hue) and object (brightness), as the visual alignment check between the real scene and the snapshot.

Frames: element positions are world metres; pixels are full-resolution ``(u, v)`` of the camera. ``project_elements`` is
pure NumPy; ``draw_overlay`` draws dots with OpenCV onto a display-scaled pane (``layout`` maps view -> display)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.common.geometry import project
from livefire.scene.snapshot import SceneSnapshot

# RGB per material class index (MATERIAL_CLASSES order: paper, cloth, wood, plastic, metal, glass, ceramic, other)
MATERIAL_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 235, 90), (255, 90, 200), (255, 150, 40), (80, 200, 255), (170, 170, 255), (90, 255, 230), (255, 120, 120),
    (110, 255, 110),
)  # fmt: skip
SELECTED_COLOR = (0, 255, 255)


def project_elements(snap: SceneSnapshot, cam: int, visible_only: bool = True) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Pixel positions of the elements seen by camera ``cam``: ``(uv (K,2) full-res px, idx (K,) element indices)``.
    Elements outside the image, behind the camera or (``visible_only``) hidden behind a nearer surface of the frozen
    ``view_depth`` (more than ``2 * surfel_spacing`` farther than it) are dropped."""
    calib = snap.cameras[cam]
    uv, z, ok = project(calib, snap.pos.astype(np.float64))
    u, v = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok = ok & (u >= 0) & (u < calib.width) & (v >= 0) & (v < calib.height)
    if visible_only and cam in snap.view_depth:
        d = snap.view_depth[cam][np.where(ok, v, 0), np.where(ok, u, 0)]
        ok = ok & ((d <= 0) | (z <= d + 2.0 * snap.surfel_spacing))
    idx = np.flatnonzero(ok)
    return uv[idx], idx


def element_colors(snap: SceneSnapshot, idx: NDArray[np.intp], material_id: NDArray[Any] | None = None,
                   selected: int | None = None) -> NDArray[np.uint8]:
    """RGB colour (K,3) per element: material colour, brightness varied by object id; the selected object is cyan."""
    mat = snap.material_id if material_id is None else material_id
    base = np.asarray(MATERIAL_COLORS, np.float32)[np.asarray(mat)[idx].astype(int) % len(MATERIAL_COLORS)]
    obj = snap.obj_id[idx].astype(int)
    shade = (0.65 + 0.35 * ((obj * 37) % 5) / 4.0).astype(np.float32)[:, None]
    out = np.clip(base * shade, 0, 255).astype(np.uint8)
    if selected is not None:
        out[obj == selected] = SELECTED_COLOR
    return out


def draw_overlay(img: NDArray[np.uint8], snap: SceneSnapshot, cam: int, pane: int, layout: Any,
                 material_id: NDArray[Any] | None = None, selected: int | None = None, radius_px: int = 1) -> int:
    """Draw the overlay in place on the display image ``img`` (RGB) for the pane ``pane`` of ``layout``
    (``ViewLayout``). Returns the number of dots drawn."""
    uv, idx = project_elements(snap, cam)
    cols = element_colors(snap, idx, material_id, selected)
    for (u, v), c in zip(uv, cols):
        x, y = layout.from_view(pane, float(u), float(v))
        cv2.circle(img, (round(x), round(y)), radius_px, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    return len(idx)
