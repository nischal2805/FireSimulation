"""Char / removed overlay: per-view disc splat of per-element ``char`` and ``removed`` (PRD 7.4).

Reuses the Task 6 disc splat (``livefire.scene.splat.iter_splat_pixels``). A splat candidate is
accepted only where ``|z_elem - view_depth| < depth_tol`` so elements hidden behind nearer surfaces do not
paint through. Coplanar elements of different objects (e.g. paper lying on the table) can both be accepted at a
pixel; the pixel is therefore owned by the object (``obj_id``) of the accepted candidate nearest to ``view_depth``
(ties: lowest obj_id) and only that object's candidates are averaged. Isolated gaps (pixels no disc reached)
inside a single-owner neighbourhood are filled by a normalised 5x5 box convolution (value*coverage / coverage).
The pixel/element association depends only on the static snapshot, so it is built once per camera; each frame is
two weighted bincounts plus a blur. Host NumPy; read-only on the fire fields.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from livefire.common.geometry import CameraCalibration
from livefire.scene.snapshot import SceneSnapshot
from livefire.scene.splat import iter_splat_pixels

FILL_RADIUS = 2  # px; normalised-convolution window is (2r+1)^2
FILL_MIN_COVERAGE = 0.6  # a gap is filled only if >= this fraction of its window is covered


class OverlayMap:
    """Static per-camera map from accepted splat candidates to pixels."""

    def __init__(self, snap: SceneSnapshot, calib: CameraCalibration, view_depth: NDArray, depth_tol: float) -> None:
        self.h, self.w = calib.height, calib.width
        n = self.h * self.w
        vd = np.asarray(view_depth, dtype=np.float64).ravel()
        e_all, p_all, dz_all = [], [], []
        for eidx, flat, z in iter_splat_pixels(calib, snap.pos, snap.area, snap.normal):
            dz = np.abs(z - vd[flat])
            ok = (vd[flat] > 0.0) & (dz < depth_tol)
            e_all.append(eidx[ok])
            p_all.append(flat[ok])
            dz_all.append(dz[ok])
        elem = np.concatenate(e_all) if e_all else np.zeros(0, np.int64)
        pix = np.concatenate(p_all) if p_all else np.zeros(0, np.int64)
        dz = np.concatenate(dz_all) if dz_all else np.zeros(0)
        oid = snap.obj_id[elem].astype(np.int64)
        # owner per pixel: candidate nearest to view_depth, ties to the lowest obj_id
        order = np.lexsort((oid, dz, pix))
        first = np.ones(len(order), dtype=bool)
        first[1:] = pix[order][1:] != pix[order][:-1]
        owner = np.full(n, -2, np.int64)  # -2: not covered
        owner[pix[order][first]] = oid[order][first]
        # reassign isolated pixels whose owner disagrees with a clear local majority (e.g. table surfels
        # showing through gaps in a sheet of paper); the pixel is then refilled from its neighbours below
        kb = (2 * FILL_RADIUS + 1,) * 2
        o2 = owner.reshape(self.h, self.w)
        best_frac = np.zeros((self.h, self.w), np.float32)
        maj = o2.copy()
        for o in np.unique(o2[o2 >= 0]):
            f = cv2.blur((o2 == o).astype(np.float32), kb, borderType=cv2.BORDER_CONSTANT)
            upd = f > best_frac
            best_frac = np.where(upd, f, best_frac)
            maj = np.where(upd, o, maj)
        owner = np.where((o2 >= 0) & (best_frac >= FILL_MIN_COVERAGE), maj, o2).ravel()
        keep = oid == owner[pix]
        self.elem, self.pix = elem[keep], pix[keep]
        cnt = np.bincount(self.pix, minlength=n).astype(np.float32)
        self.inv_count = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1.0), 0.0).astype(np.float32)
        self.cov = (cnt > 0).reshape(self.h, self.w).astype(np.float32)
        # gaps that may be filled: uncovered, valid depth, mostly-covered single-owner neighbourhood
        k = np.ones((2 * FILL_RADIUS + 1,) * 2, np.uint8)
        own = np.where(cnt > 0, owner, -2).reshape(self.h, self.w).astype(np.float32)
        omin = cv2.erode(np.where(self.cov > 0, own, 1e9).astype(np.float32), k)
        omax = cv2.dilate(np.where(self.cov > 0, own, -1e9).astype(np.float32), k)
        frac = cv2.blur(self.cov, k.shape, borderType=cv2.BORDER_CONSTANT)
        self.fill = (self.cov == 0) & (frac >= FILL_MIN_COVERAGE) & (omin == omax) & (vd.reshape(self.h, self.w) > 0)
        self._den = cv2.blur(self.cov, k.shape, borderType=cv2.BORDER_CONSTANT)

    def splat(self, values: NDArray) -> NDArray[np.float32]:
        """Per-element values (N_e,) in 0..1 -> per-pixel mean over the owning object's accepted elements
        (H,W); isolated gaps filled by normalised convolution; 0 elsewhere."""
        v = np.asarray(values, dtype=np.float64)[self.elem]
        acc = np.bincount(self.pix, weights=v, minlength=self.h * self.w).astype(np.float32)
        out = (acc * self.inv_count).reshape(self.h, self.w)
        if self.fill.any():
            num = cv2.blur(out, (2 * FILL_RADIUS + 1,) * 2,
                           borderType=cv2.BORDER_CONSTANT)  # value*cov (out is 0 where uncovered)
            out = np.where(self.fill, num / np.maximum(self._den, 1e-6), out)
        return out.astype(np.float32)
