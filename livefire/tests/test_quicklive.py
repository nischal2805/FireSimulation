"""Plane-derived world for the uncalibrated quick live mode (synthetic ground truth, no models)."""

import numpy as np

from livefire.app.quicklive import fit_plane_ransac, intrinsics_from_fov, pixel_rays, world_from_plane
from livefire.common.geometry import CameraCalibration, backproject, project


def test_plane_world_recovers_table_and_height():
    w, h = 320, 180
    K = intrinsics_from_fov(w, h, 65.0)
    # camera 0.6 m above z=0 table, pitched 40 deg down, looking along +Y world
    pitch = np.radians(40.0)
    fwd = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], axis=1)
    T[:3, 3] = [0.0, 0.0, 0.6]
    rays = pixel_rays(K, h, w)
    d_w = rays @ T[:3, :3].T  # world ray directions
    t = -0.6 / np.where(d_w[..., 2] < -1e-3, d_w[..., 2], -1e-3)
    z = np.where(d_w[..., 2] < -1e-3, t, 0.0)  # table depth (rays that hit it)
    valid = z > 0
    n, d, mask = fit_plane_ransac(z, rays, valid, up_pref=np.array([0.0, -1.0, 0.0]))
    assert mask.sum() > 0.9 * valid.sum()
    cen = (rays[mask] * z[mask][:, None]).mean(axis=0)
    T_wc = world_from_plane(n, cen - (n @ cen - d) * n)
    cal = CameraCalibration(0, w, h, K, np.zeros(5), T_wc, 1)
    pts = backproject(cal, np.array([[w / 2, h / 2]]), np.array([z[h // 2, w // 2]]))
    assert abs(pts[0, 2]) < 1e-3  # table points land on z = 0
    assert T_wc[2, 3] > 0  # camera above the plane
    assert abs(T_wc[2, 3] - 0.6) < 0.02  # and at the right height
    uv, _zc, ok = project(cal, pts)
    assert ok[0] and np.allclose(uv[0], [w / 2, h / 2], atol=1e-3)
