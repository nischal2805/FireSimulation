"""Per-view depth accumulation over the acquisition window (PRD 7.2).

Frame: camera OpenCV, Z-depth metres, invalid = 0.0 with ``valid = False`` (PRD 4). Host NumPy, caller-owned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class AccumulatedDepth:
    """Robust per-pixel depth of one view. ``depth`` f32 (H,W) m (0.0 where invalid), ``valid`` bool (H,W),
    ``conf`` f32 (H,W) in [0,1] = fraction of the ``n_frames`` frames that are valid and agree with the median
    within the relative tolerance (0 where invalid), ``n_frames`` the number of frames accumulated."""

    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    conf: NDArray[np.float32]
    n_frames: int


def accumulate_depth(
    depths: NDArray, valid: NDArray, agree_tol_rel: float = 0.02, min_confidence: float = 0.5, row_chunk: int = 64
) -> AccumulatedDepth:
    """Median over frames of the valid depths per pixel.

    ``depths`` (T,H,W) Z-depth m, ``valid`` (T,H,W) bool. The median is taken over the valid samples of a pixel (mean
    of the two middle samples for an even count). A frame *agrees* if ``|d - median| <= agree_tol_rel * median``;
    ``conf = agreeing / T`` (so pixels missing in many frames get a low confidence). Pixels with ``conf <
    min_confidence`` are invalid (depth 0.0). Processed in row chunks to bound memory."""
    depths = np.asarray(depths)
    valid = np.asarray(valid, dtype=bool)
    if depths.ndim != 3 or valid.shape != depths.shape:
        raise ValueError(f"expected (T,H,W) depths and valid, got {depths.shape} {valid.shape}")
    t, h, w = depths.shape
    if t < 1:
        raise ValueError("need at least one frame")
    med = np.zeros((h, w), dtype=np.float32)
    conf = np.zeros((h, w), dtype=np.float32)
    for r0 in range(0, h, row_chunk):
        sl = slice(r0, min(h, r0 + row_chunk))
        d = depths[:, sl].astype(np.float32)
        v = valid[:, sl] & (d > 0.0)
        s = np.sort(np.where(v, d, np.inf), axis=0)
        k = v.sum(axis=0)
        lo = np.take_along_axis(s, np.clip((k - 1) // 2, 0, t - 1)[None], axis=0)[0]
        hi = np.take_along_axis(s, np.clip(k // 2, 0, t - 1)[None], axis=0)[0]
        m = np.where(k > 0, 0.5 * (lo + hi), 0.0)
        agree = v & (np.abs(d - m[None]) <= agree_tol_rel * m[None])
        med[sl] = m
        conf[sl] = agree.sum(axis=0) / t
    ok = (conf >= min_confidence) & (med > 0.0)
    return AccumulatedDepth(
        np.where(ok, med, 0.0).astype(np.float32), ok, np.where(ok, conf, 0.0).astype(np.float32), t
    )
