"""Perception result types (depth, segmentation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class DepthResult:
    """Per-frame depth. Z-depth along camera +Z in metres (not ray distance).

    ``z_depth_m`` f32 (H,W); ``valid`` bool (H,W); ``conf`` f32 (H,W) in [0,1].
    Invalid pixels hold exactly 0.0 with ``valid=False``; NaN is never allowed.
    ``calib_version`` is that of the input frame.
    """

    z_depth_m: np.ndarray
    valid: np.ndarray
    conf: np.ndarray
    calib_version: int

    def __post_init__(self) -> None:
        z, v, c = self.z_depth_m, self.valid, self.conf
        if z.dtype != np.float32 or z.ndim != 2:
            raise ValueError(f"z_depth_m must be float32 (H,W), got {z.dtype} {z.shape}")
        if v.dtype != np.bool_ or v.shape != z.shape:
            raise ValueError(f"valid must be bool {z.shape}, got {v.dtype} {v.shape}")
        if c.dtype != np.float32 or c.shape != z.shape:
            raise ValueError(f"conf must be float32 {z.shape}, got {c.dtype} {c.shape}")
        if np.isnan(z).any() or np.isnan(c).any():
            raise ValueError("depth/conf contain NaN; invalid pixels must be 0.0")
        if (z[~v] != 0.0).any():
            raise ValueError("invalid pixels must hold depth 0.0")
        if (c < 0.0).any() or (c > 1.0).any():
            raise ValueError("conf must lie in [0,1]")


@dataclass(frozen=True, eq=False)
class SegmentationResult:
    """Instance segmentation of one frame.

    ``instance_ids`` int32 (H,W): per-camera instance id, -1 = background/unknown (an
    observation id, not a world ``obj_id``). ``categories`` maps instance id -> category
    label. ``scores`` maps instance id -> confidence in [0,1]. ``calib_version`` is that of
    the input frame.
    """

    instance_ids: np.ndarray
    categories: dict[int, str]
    scores: dict[int, float]
    calib_version: int

    def __post_init__(self) -> None:
        a = self.instance_ids
        if a.dtype != np.int32 or a.ndim != 2:
            raise ValueError(f"instance_ids must be int32 (H,W), got {a.dtype} {a.shape}")
