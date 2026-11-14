"""Geometry contract tests (PRD section 4). No camera, no AI."""

import numpy as np
import pytest

from livefire.common.geometry import (
    AABB,
    CameraCalibration,
    VoxelGrid,
    backproject,
    backproject_depth,
    invert_transform,
    look_at,
    make_K,
    pixel_grid,
    project,
    ray_dirs_camera,
    raydist_to_zdepth,
    transform_points,
    zdepth_to_raydist,
)

W, H = 1280, 720


def _roll(T: np.ndarray, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ Rz  # roll about the camera's own optical axis
    return out


def _calib(cam_id: int, T: np.ndarray, K: np.ndarray) -> CameraCalibration:
    return CameraCalibration(cam_id, W, H, K, np.zeros(5), T, 1)


@pytest.fixture
def cams() -> list[CameraCalibration]:
    c0 = _calib(0, look_at((0.9, -1.1, 0.8), (0.0, 0.0, 0.5)), make_K(900.0, 905.0, 655.3, 341.9))
    T1 = _roll(look_at((-1.2, 0.7, 1.1), (0.0, 0.0, 0.5)), 17.0)
    c1 = _calib(1, T1, make_K(870.0, 868.0, 640.7, 352.2))
    return [c0, c1]


def test_roundtrip_two_asymmetric_cameras(cams):
    rng = np.random.default_rng(1234)
    pts = rng.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], size=(200, 3)) + [0.0, 0.0, 0.0]
    for cam in cams:
        uv, z, valid = project(cam, pts)
        assert valid.all()
        assert (np.abs(uv - [655.3, 341.9]).max(axis=0) > 1.0).all()  # genuinely off-axis
        back = backproject(cam, uv, z)
        assert np.abs(back - pts).max() < 1e-4


def test_transform_direction_flip_fails(cams):
    rng = np.random.default_rng(7)
    pts = rng.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], size=(200, 3))
    cam = cams[1]
    uv, z, _ = project(cam, pts)
    wrong = CameraCalibration(
        cam.cam_id, W, H, cam.K, cam.dist, cam.T_camera_from_world, cam.calib_version
    )
    assert np.abs(backproject(cam, uv, z) - pts).max() < 1e-4
    assert np.abs(backproject(wrong, uv, z) - pts).max() > 0.1


def test_look_at_hand_computed():
    # Camera at (0,0,1) looking at the origin (top-down; look_at falls back to up=+Y).
    K = make_K(800.0, 800.0, 640.0, 360.0)
    cam = _calib(0, look_at((0, 0, 1), (0, 0, 0)), K)
    uv, z, valid = project(cam, np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]))
    assert valid.all() and np.allclose(z, 1.0)
    assert np.allclose(uv[0], [640.0, 360.0])  # origin -> (cx, cy)
    assert np.allclose(uv[1], [640.0 + 80.0, 360.0])  # world +X -> image +u, hand: fx*0.1/1
    assert np.allclose(uv[2], [640.0, 360.0 - 80.0])  # world +Y -> image up (-v)


def test_look_at_world_x_maps_to_image_u():
    K = make_K(800.0, 800.0, 640.0, 360.0)
    # Camera at (0,-2,1) looking at the origin: world +X is image right, world +Z is image up.
    cam = _calib(0, look_at((0, -2, 1), (0, 0, 0)), K)
    uv, z, valid = project(cam, np.array([[0, 0, 0], [0.1, 0, 0], [0, 0, 0.1]], dtype=float))
    assert valid.all()
    assert np.allclose(uv[0], [640.0, 360.0])
    assert uv[1, 0] > 640.0 and np.isclose(uv[1, 1], 360.0)
    assert uv[2, 1] < 360.0  # +Z up -> smaller v (OpenCV +Y down)
    # exact: camera z axis = (0,2,-1)/sqrt5, x axis = (1,0,0); point (0.1,0,0) -> u = cx + fx*0.1/depth
    depth = np.sqrt(5.0)
    assert np.isclose(z[0], depth)
    assert np.isclose(uv[1, 0], 640.0 + 800.0 * 0.1 / depth)


def test_project_behind_camera_is_invalid_and_finite(cams):
    cam = cams[0]
    behind = cam.center_world + 1.0 * (cam.center_world - np.array([0.0, 0.0, 0.5]))
    uv, z, valid = project(cam, behind[None])
    assert not valid[0] and z[0] <= 0 and np.all(uv == 0.0) and np.isfinite(uv).all()


def test_zdepth_vs_raydist():
    K = make_K(900.0, 900.0, 655.3, 341.9)
    cam = _calib(0, look_at((0.5, -1.0, 0.7), (0, 0, 0.3)), K)
    rays = ray_dirs_camera(cam)
    assert rays.shape == (H, W, 3) and np.allclose(rays[..., 2], 1.0)
    z = np.full((H, W), 1.3)
    r = zdepth_to_raydist(cam, z)
    corner = (0, 0)
    expected = np.linalg.norm(np.linalg.inv(K) @ [0.0, 0.0, 1.0])
    assert np.isclose(r[corner] / z[corner], expected)
    assert r[corner] > z[corner]
    assert np.isclose(r[341, 655] / z[341, 655], 1.0, atol=1e-3)  # near principal point
    assert np.allclose(raydist_to_zdepth(cam, r), z)
    z0 = z.copy()
    z0[10, 10] = 0.0
    assert zdepth_to_raydist(cam, z0)[10, 10] == 0.0
    assert raydist_to_zdepth(cam, zdepth_to_raydist(cam, z0))[10, 10] == 0.0


def test_backproject_depth_matches_pointwise_and_invalid_zero(cams):
    cam = cams[0]
    u, v = pixel_grid(W, H)
    assert u[0, 5] == 5 and v[3, 0] == 3
    depth = np.full((H, W), 1.5)
    valid = np.ones((H, W), dtype=bool)
    valid[100:110, 200:210] = False
    pts = backproject_depth(cam, depth, valid)
    assert pts.shape == (H, W, 3)
    assert np.all(pts[100:110, 200:210] == 0.0)
    one = backproject(cam, np.array([[300.0, 400.0]]), np.array([1.5]))[0]
    assert np.allclose(pts[400, 300], one)
    _, z, _ok = project(cam, pts[valid][:50])
    assert np.allclose(z, 1.5, atol=1e-9)


def test_calibration_rejects_bad_rotation_and_shapes():
    K = make_K(800.0, 800.0, 640.0, 360.0)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, 1.0, 1.2])
    with pytest.raises(ValueError):
        _calib(0, T, K)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, 1.0, -1.0])  # reflection, det -1
    with pytest.raises(ValueError):
        _calib(0, T, K)
    with pytest.raises(ValueError):
        CameraCalibration(0, W, H, np.eye(4), np.zeros(5), np.eye(4), 1)
    with pytest.raises(ValueError):
        CameraCalibration(0, W, H, K, np.zeros(4), np.eye(4), 1)


def test_invert_and_transform_points(cams):
    T = cams[1].T_world_from_camera
    assert np.allclose(invert_transform(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(cams[1].T_camera_from_world, invert_transform(T))
    assert np.allclose(cams[1].center_world, T[:3, 3])
    p = np.array([[0.1, 0.2, 0.3]])
    assert np.allclose(transform_points(invert_transform(T), transform_points(T, p)), p)


def test_voxel_roundtrip_corners_interior_and_min_edge():
    roi = AABB(np.array([-1.5, -1.5, -0.1]), np.array([1.5, 1.5, 2.9]))
    g = VoxelGrid(roi, 64)
    assert np.isclose(g.h, 3.0 / 64)
    n = g.N - 1
    corners = np.array([[i, j, k] for i in (0, n) for j in (0, n) for k in (0, n)])
    assert np.array_equal(g.index_of(g.centers(corners)), corners)
    assert np.allclose(g.centers(np.array([[0, 0, 0]]))[0], roi.min + 0.5 * g.h)
    rng = np.random.default_rng(5)
    idx = rng.integers(0, g.N, size=(500, 3))
    assert np.array_equal(g.index_of(g.centers(idx)), idx)
    p = rng.uniform(roi.min, roi.max - 1e-9, size=(500, 3))
    idx = g.index_of(p)
    assert g.in_bounds(idx).all()
    assert np.all(np.abs(g.centers(idx) - p) <= g.h / 2 + 1e-12)
    assert np.array_equal(g.index_of(roi.min[None]), [[0, 0, 0]])
    assert not g.in_bounds(g.index_of(np.array([[-1.6, 0, 0], [1.5, 0, 0]]))).any()
    lin = g.linear_index(idx)
    assert np.array_equal(g.unravel(lin), idx)
    assert g.linear_index(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])).tolist() == [1, 64, 4096]


def test_voxel_grid_requires_cubic_roi():
    with pytest.raises(ValueError):
        VoxelGrid(AABB(np.array([0.0, 0, 0]), np.array([1.0, 1.0, 2.0])), 8)


def test_aabb_basics_and_ray_intersect():
    box = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    assert np.allclose(box.size, 2.0) and np.allclose(box.center, 0.0)
    assert box.contains(np.array([[0, 0, 0], [1, 1, 1], [1.1, 0, 0]], dtype=float)).tolist() == [
        True,
        True,
        False,
    ]
    o = np.array([[-3, 0, 0], [-3, 2, 0], [0, 0, 0], [3, 0, 0], [-3, 0, 0], [0, 5, 0]], float)
    d = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    t0, t1, hit = box.ray_intersect(o, d)
    assert hit.tolist() == [True, False, True, False, False, False]
    assert np.isclose(t0[0], 2.0) and np.isclose(t1[0], 4.0)
    assert t0[2] == 0.0 and np.isclose(t1[2], 1.0)  # origin inside
    assert np.isfinite(t0).all() and np.isfinite(t1).all()
    # oblique hit: ray through the corner region
    o2 = np.array([[-2.0, -2.0, 0.0]])
    d2 = np.array([[1.0, 1.0, 0.0]])
    a, b, h = box.ray_intersect(o2, d2)
    assert h[0] and np.isclose(a[0], 1.0) and np.isclose(b[0], 3.0)
