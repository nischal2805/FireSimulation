"""TSDF fusion in Warp over a sub-ROI (PRD 6.1, 7.2).

Frames: world (metres, +Z up), camera OpenCV. Voxel centre = ``roi_min + (i + 0.5) * h`` (PRD 4; the same formula as
``common.geometry.VoxelGrid``, which is cubic-only, so the non-cubic fusion ROI uses ``TSDFGrid`` with per-axis counts).
Device: Warp device from ``configs/app.yaml`` (``cpu`` here, ``cuda:0`` on the GPU machine). The volume owns its device
arrays; ``tsdf()`` / ``weight()`` return host copies owned by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from numpy.typing import NDArray

from livefire.common.geometry import AABB, CameraCalibration
from livefire.sim.device import select_device


@dataclass(frozen=True)
class TSDFGrid:
    """Voxel grid over an AABB: ``n = ceil(size / h)`` per axis; the grid covers ``roi_min .. roi_min + n h`` (it may
    extend up to one voxel past ``roi.max``)."""

    roi: AABB
    h: float

    @property
    def shape(self) -> tuple[int, int, int]:
        n = np.ceil(self.roi.size / self.h - 1e-9).astype(int)
        return int(n[0]), int(n[1]), int(n[2])

    def centers(self, idx: NDArray) -> NDArray[np.float64]:
        """Integer indices (M,3) -> voxel centres (M,3) m."""
        return self.roi.min + (np.atleast_2d(idx) + 0.5) * self.h


@wp.kernel
def _integrate(
    tsdf: wp.array3d(dtype=wp.float32),
    weight: wp.array3d(dtype=wp.float32),
    depth: wp.array2d(dtype=wp.float32),
    conf: wp.array2d(dtype=wp.float32),
    fx: float, fy: float, cx: float, cy: float,
    T_cw: wp.mat44,
    roi_min: wp.vec3,
    h: float,
    trunc: float,
    max_depth: float,
    min_z: float,
):  # fmt: skip
    i, j, k = wp.tid()
    p = roi_min + wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h)
    pc = wp.transform_point(T_cw, p)
    z = pc[2]
    if z < min_z:
        return
    u = fx * pc[0] / z + cx
    v = fy * pc[1] / z + cy
    ui = int(wp.floor(u + 0.5))
    vi = int(wp.floor(v + 0.5))
    if ui < 0 or vi < 0 or ui >= depth.shape[1] or vi >= depth.shape[0]:
        return
    d = depth[vi, ui]
    c = conf[vi, ui]
    if d <= 0.0 or c <= 0.0 or d > max_depth:
        return
    # signed distance along the viewing ray (positive in front of the surface = free space)
    sdf = (d - z) * wp.length(pc) / z
    if sdf < -trunc:
        return  # behind the surface by more than the truncation: occluded, no information
    obs = wp.min(1.0, sdf / trunc)
    w0 = weight[i, j, k]
    w1 = w0 + c
    tsdf[i, j, k] = (tsdf[i, j, k] * w0 + obs * c) / w1
    weight[i, j, k] = w1


class TSDFVolume:
    """Weighted-average truncated signed distance volume (positive = free space in front of the surface).

    ``integrate`` fuses one anchored view; voxels never observed keep weight 0. Truncation ``trunc_voxels * h``."""

    def __init__(self, roi: AABB, voxel_size: float, trunc_voxels: float = 3.0, device: str | None = None) -> None:
        self.device = select_device(device)
        self.grid = TSDFGrid(roi, float(voxel_size))
        self.trunc = float(trunc_voxels) * float(voxel_size)
        shp = self.grid.shape
        self._tsdf = wp.zeros(shp, dtype=wp.float32, device=self.device)
        self._w = wp.zeros(shp, dtype=wp.float32, device=self.device)
        self.n_views = 0

    def integrate(
        self, calib: CameraCalibration, depth: NDArray, conf: NDArray, max_depth: float = 20.0
    ) -> None:
        """Fuse a view: ``depth`` f32 (H,W) Z-depth m (0 = invalid), ``conf`` f32 (H,W) in [0,1] used as weight."""
        if depth.shape != (calib.height, calib.width) or conf.shape != depth.shape:
            raise ValueError(f"depth/conf {depth.shape} {conf.shape} do not match calibration")
        d = wp.array(np.ascontiguousarray(depth, dtype=np.float32), dtype=wp.float32, device=self.device)
        c = wp.array(np.ascontiguousarray(conf, dtype=np.float32), dtype=wp.float32, device=self.device)
        K = calib.K
        wp.launch(
            _integrate, dim=self.grid.shape,
            inputs=[self._tsdf, self._w, d, c, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                    wp.mat44(*calib.T_camera_from_world.astype(np.float32).ravel()),
                    wp.vec3(*self.grid.roi.min.astype(np.float32)), float(self.grid.h), self.trunc,
                    float(max_depth), 1e-3],
            device=self.device,
        )  # fmt: skip
        wp.synchronize_device(self.device)
        self.n_views += 1

    def tsdf(self) -> NDArray[np.float32]:
        """Host copy of the TSDF (nx,ny,nz) in [-1, 1] (0 where weight is 0)."""
        return self._tsdf.numpy()

    def weight(self) -> NDArray[np.float32]:
        """Host copy of the accumulated weights (nx,ny,nz)."""
        return self._w.numpy()
