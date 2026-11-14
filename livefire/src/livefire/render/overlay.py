"""Char / removed overlay: per-view disc splat of per-element ``char`` and ``removed`` (PRD 7.4).

Reuses the Task 6 disc splat (``livefire.scene.splat.iter_splat_pixels``). A splat candidate is
accepted only where ``|z_elem - view_depth| < depth_tol`` so elements hidden behind nearer surfaces do not
paint through. The pixel/element association depends only on the static snapshot, so it is built once
per camera; each frame is two weighted bincounts. Host NumPy; read-only on the fire fields.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from livefire.common.geometry import CameraCalibration
from livefire.scene.snapshot import SceneSnapshot
from livefire.scene.splat import iter_splat_pixels


class OverlayMap:
    """Static per-camera map from accepted splat candidates to pixels."""

    def __init__(self, snap: SceneSnapshot, calib: CameraCalibration, view_depth: NDArray, depth_tol: float) -> None:
        self.h, self.w = calib.height, calib.width
        vd = np.asarray(view_depth, dtype=np.float64).ravel()
        e_all, p_all = [], []
        for eidx, flat, z in iter_splat_pixels(calib, snap.pos, snap.area, snap.normal):
            ok = (vd[flat] > 0.0) & (np.abs(z - vd[flat]) < depth_tol)
            e_all.append(eidx[ok])
            p_all.append(flat[ok])
        self.elem = np.concatenate(e_all) if e_all else np.zeros(0, np.int64)
        self.pix = np.concatenate(p_all) if p_all else np.zeros(0, np.int64)
        cnt = np.bincount(self.pix, minlength=self.h * self.w).astype(np.float32)
        self.inv_count = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1.0), 0.0).astype(np.float32)

    def splat(self, values: NDArray) -> NDArray[np.float32]:
        """Per-element values (N_e,) in 0..1 -> per-pixel mean over accepted elements (H,W); 0 elsewhere."""
        v = np.asarray(values, dtype=np.float64)[self.elem]
        acc = np.bincount(self.pix, weights=v, minlength=self.h * self.w).astype(np.float32)
        return (acc * self.inv_count).reshape(self.h, self.w)
