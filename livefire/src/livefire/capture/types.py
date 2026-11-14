"""Capture data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Frame:
    """One camera frame.

    ``rgb``: uint8 (H, W, 3), undistorted sRGB, host (CPU) memory, read-only by convention
    (the producer hands over ownership; consumers must copy before mutating).
    ``t_arrive_ns``: monotonic arrival time (ns). ``seq``: per-camera sequence number.
    ``calib_version``: calibration the undistortion used (PRD 4); consumers must check it.
    """

    cam_id: int
    rgb: np.ndarray
    t_arrive_ns: int
    seq: int
    calib_version: int

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must be uint8 (H,W,3), got {self.rgb.dtype} {self.rgb.shape}")
