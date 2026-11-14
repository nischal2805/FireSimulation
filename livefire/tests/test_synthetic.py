"""Synthetic scene + camera source tests (P0-005)."""

import time

import numpy as np
import pytest

from livefire.capture.synthetic_source import SyntheticCameraSource
from livefire.common.geometry import backproject, project
from livefire.common.interfaces import CameraSource
from livefire.scene.synthetic import (
    Box,
    build_table_v1,
    load_cameras,
    render_view,
    validate_materials,
)


@pytest.fixture(scope="module")
def rig():
    scene = build_table_v1()
    cams = load_cameras()
    views = {i: render_view(scene, c, seed=0) for i, c in cams.items()}
    return scene, cams, views


def test_materials_valid(rig):
    validate_materials(rig[0])
    assert rig[0].objects_table()[3] == ("paper_sheet", "paper")


def test_backprojected_points_on_primitive(rig):
    scene, cams, views = rig
    rng = np.random.default_rng(7)
    for i, v in views.items():
        vs, us = np.nonzero(v.valid & (v.obj_id >= 0))
        sel = rng.choice(len(vs), 500, replace=False)
        uv = np.stack([us[sel], vs[sel]], axis=1).astype(float)
        pts = backproject(cams[i], uv, v.depth[vs[sel], us[sel]])
        ids = v.obj_id[vs[sel], us[sel]]
        for oid in np.unique(ids):
            m = ids == oid
            assert scene.distance_to_object(int(oid), pts[m]).max() < 1e-4


def test_depth_equals_analytic_intersection(rig):
    """Analytic Z-depth for the table top (plane z=0) and the paper sheet (plane y=0)."""
    _, cams, views = rig
    for i, v in views.items():
        c = cams[i]
        R, o = c.T_world_from_camera[:3, :3], c.center_world
        # a pixel on the table top away from objects: centre-ish ray towards (0.6,-0.3,0)
        uv, _, _ = project(c, np.array([[0.6, -0.3, 0.0]]))
        u, vv = round(float(uv[0, 0])), round(float(uv[0, 1]))
        assert v.obj_id[vv, u] == 0
        d = np.array([u, vv, 1.0]) @ np.linalg.inv(c.K).T @ R.T
        t = (0.0 - o[2]) / d[2]  # plane z = 0, d has camera z = 1 so t is Z-depth
        assert abs(v.depth[vv, u] - t) < 1e-4
        # paper: y = 0
        m = v.obj_id == 3
        vs, us = np.nonzero(m)
        k = len(vs) // 2
        d = np.array([us[k], vs[k], 1.0]) @ np.linalg.inv(c.K).T @ R.T
        assert abs(v.depth[vs[k], us[k]] - (0.0 - o[1]) / d[1]) < 1e-4


def test_all_objects_visible_both_views(rig):
    for v in rig[2].values():
        for oid in range(5):
            assert (v.obj_id == oid).sum() >= 200, oid


def test_miss_pixels_and_room(rig):
    for v in rig[2].values():
        assert np.isfinite(v.depth).all()
        assert (v.depth[~v.valid] == 0).all() and (v.rgb[~v.valid] == 0).all()
        assert (v.obj_id[~v.valid] == -1).all()
        assert v.valid.mean() > 0.99  # camera is inside the closed room
        assert v.rgb.dtype == np.uint8 and v.rgb_linear.dtype == np.float32


def test_determinism_and_noise(rig):
    scene, cams, _ = rig
    a, b = render_view(scene, cams[0], seed=3), render_view(scene, cams[0], seed=3)
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.depth, b.depth)
    n1 = render_view(scene, cams[0], seed=3, noise_sigma=0.02)
    n2 = render_view(scene, cams[0], seed=3, noise_sigma=0.02)
    n3 = render_view(scene, cams[0], seed=4, noise_sigma=0.02)
    assert np.array_equal(n1.rgb, n2.rgb) and not np.array_equal(n1.rgb, n3.rgb)
    assert not np.array_equal(n1.rgb, a.rgb)


def test_sample_surface_and_area():
    scene = build_table_v1()
    rng = np.random.default_rng(1)
    pts, nrm, area = scene.sample_surface(1, 400, rng)
    assert abs(area - 2 * (0.15 * 0.1 + 0.15 * 0.08 + 0.1 * 0.08)) < 1e-12
    assert scene.distance_to_object(1, pts).max() < 1e-12
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    # offsetting along the outward normal gives distance = offset
    assert np.allclose(scene.distance_to_object(1, pts + 0.01 * nrm), 0.01)
    p2, _, a2 = scene.sample_surface(3, 50, np.random.default_rng(1))
    assert abs(a2 - 0.15 * 0.2) < 1e-12 and scene.distance_to_object(3, p2).max() < 1e-12
    # same seed, same samples
    assert np.array_equal(scene.sample_surface(1, 400, np.random.default_rng(1))[0], pts)
    assert isinstance(scene.objects[0], Box)


def test_source_latest_increasing_and_drops_stale(rig):
    _, cams, views = rig
    src = SyntheticCameraSource(0, views[0].rgb, cams[0].calib_version, fps=200)
    assert isinstance(src, CameraSource)
    assert src.latest() is None
    src.start()
    try:
        f1 = src.latest()
        assert f1 is not None and f1.calib_version == cams[0].calib_version
        time.sleep(0.1)  # not consuming: stale frames must be dropped
        f2 = src.latest()
        assert f2.seq > f1.seq and f2.t_arrive_ns > f1.t_arrive_ns
        assert f2.seq - f1.seq >= 2  # skipped, not queued
        time.sleep(0.03)
        assert src.latest().seq > f2.seq
        assert not f2.rgb.flags.writeable
    finally:
        src.stop()
    s = src.latest().seq
    time.sleep(0.05)
    assert src.latest().seq == s  # stopped
