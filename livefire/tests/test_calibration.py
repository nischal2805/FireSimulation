"""ChArUco calibration (P1-003): synthetic ground truth, headless."""

import cv2
import numpy as np
import pytest

from livefire.calib.board import (
    BoardSpec,
    T_world_from_board,
    board_corners_world,
    detect_charuco,
    load_board_spec,
    render_board_png,
)
from livefire.calib.extrinsics import estimate_extrinsics
from livefire.calib.intrinsics import CaptureGate, CoverageMap, View, fit_intrinsics, split_views
from livefire.calib.store import (
    CalibrationError,
    active_version,
    check_frame_size,
    list_versions,
    load_calibration,
    load_intrinsics,
    save_calibration,
    save_intrinsics,
)
from livefire.calib.synthetic_board import SyntheticBoardCamera, random_view_poses
from livefire.common.geometry import look_at, make_K, project

SIZE = (1280, 720)
K_TRUE = make_K(900.0, 906.0, 640.3, 358.7)
DIST_TRUE = np.array([-0.10, 0.03, 0.001, -0.0005, 0.0])
SPEC = BoardSpec()


@pytest.fixture(scope="module")
def cam():
    return SyntheticBoardCamera(SPEC, K_TRUE, DIST_TRUE, SIZE)


@pytest.fixture(scope="module")
def views(cam):
    rng = np.random.default_rng(7)
    out, imgs = [], {}
    for i, T in enumerate(random_view_poses(40, rng)):
        img = cam.render(T, rng)
        c, ids = detect_charuco(SPEC, img)
        if len(ids) >= 12:
            out.append(View(c, ids, f"v{i:02d}.png"))
            imgs[f"v{i:02d}.png"] = img
    return out, imgs


def test_config_board_matches_brief():
    s = load_board_spec()
    assert (s.squares_x, s.squares_y, s.dictionary) == (7, 5, "DICT_5X5_100")
    assert s.square_length_m == pytest.approx(0.035) and s.marker_length_m == pytest.approx(0.026)


def test_board_png_physical_size():
    img, info = render_board_png(SPEC, dpi=300)
    assert info["board_mm"] == pytest.approx((245.0, 175.0)) and info["page_mm"] == (297.0, 210.0)
    assert img.shape == (round(210 / 25.4 * 300), round(297 / 25.4 * 300))
    _c, ids = detect_charuco(SPEC, img)
    assert len(ids) == 24  # all inner corners found in the print-ready image


def test_intrinsics_recovered_and_heldout(views):
    vs, _imgs = views
    assert len(vs) >= 25
    r = fit_intrinsics(SPEC, vs, SIZE, min_views=20, holdout_fraction=0.2, seed=1)
    K = r.K_raw
    for got, want in ((K[0, 0], 900.0), (K[1, 1], 906.0), (K[0, 2], 640.3), (K[1, 2], 358.7)):
        assert abs(got - want) / want < 0.005, (got, want)
    assert abs(r.dist[0] - DIST_TRUE[0]) < 0.02
    assert r.n_heldout >= 5 and r.n_fit >= 20
    assert r.rms_heldout < 0.5 and r.median_heldout < 0.5
    assert set(r.fit_names).isdisjoint(r.heldout_names)
    assert "HELD-OUT" in r.summary()
    # undistorted K policy: alpha 0, same size; with real distortion the K differs from K_raw
    assert r.alpha == 0.0 and not np.allclose(r.K_undist, r.K_raw)


def test_split_is_seeded_and_disjoint():
    a = split_views(25, 0.2, 5)
    b = split_views(25, 0.2, 5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(set(a[0]) & set(a[1])) == 0 and len(a[1]) == 5
    with pytest.raises(ValueError):
        split_views(9, 0.2, 0)


def test_too_few_views_raises(views):
    with pytest.raises(ValueError, match="need >= 20"):
        fit_intrinsics(SPEC, views[0][:10], SIZE, min_views=20)


def _intr_truth(cam_id=0):
    from livefire.calib.intrinsics import IntrinsicsResult
    Ku, _ = cv2.getOptimalNewCameraMatrix(K_TRUE, DIST_TRUE, SIZE, 0.0, SIZE)
    return IntrinsicsResult(cam_id, SIZE[0], SIZE[1], K_TRUE, DIST_TRUE, Ku, 0.0, 0.1, 0.1, 0.1, 0.1, 20, 5)


def _rot_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def _two_cams(cam):
    rng = np.random.default_rng(3)
    poses = {0: look_at([0.55, -0.45, 0.6], [0.0, 0.0, 0.0]), 1: look_at([-0.4, -0.6, 0.55], [0.0, 0.0, 0.0])}
    v = {}
    for cid, T in poses.items():
        c, ids = detect_charuco(SPEC, cam.render(T, rng, noise_sigma=1.0))
        v[cid] = View(c, ids)
    return poses, v


def test_extrinsics_recover_pose(cam):
    poses, v = _two_cams(cam)
    res = estimate_extrinsics(SPEC, {0: _intr_truth(0), 1: _intr_truth(1)}, v)
    for c in res.cameras:
        T = poses[c.cam_id]
        assert np.linalg.norm(c.T_world_from_camera[:3, 3] - T[:3, 3]) < 0.002
        assert _rot_deg(c.T_world_from_camera[:3, :3], T[:3, :3]) < 0.2
        assert c.reproj_rms_px < 0.5
    assert res.consistency_m[(0, 1)] < 0.002 and not res.warnings


def test_world_frame_convention(cam):
    _poses, v = _two_cams(cam)
    res = estimate_extrinsics(SPEC, {0: _intr_truth(0), 1: _intr_truth(1)}, v)
    T_wb = T_world_from_board(SPEC)
    w, h = SPEC.size_m
    assert np.allclose((T_wb @ [w / 2, h / 2, 0, 1])[:3], 0)           # board centre -> origin
    assert np.allclose(T_wb[:3, 2], [0, 0, -1]) and np.linalg.det(T_wb[:3, :3]) == pytest.approx(1)
    assert np.allclose(board_corners_world(SPEC)[:, 2], 0)           # table plane z = 0
    assert np.allclose(T_wb[:3, 0], [1, 0, 0])                         # +X along board x
    for c in res.cameras:
        assert c.T_world_from_camera[2, 3] > 0.3                       # cameras above the table, +Z toward them
    # board centre projects inside the image of each camera through the stored (world) pose
    from livefire.common.geometry import CameraCalibration
    for c in res.cameras:
        cal = CameraCalibration(c.cam_id, *SIZE, _intr_truth().K_undist, np.zeros(5), c.T_world_from_camera, 1)
        uv, _z, ok = project(cal, np.zeros((1, 3)))
        assert ok[0] and 0 < uv[0, 0] < SIZE[0] and 0 < uv[0, 1] < SIZE[1]


def test_extrinsics_too_few_corners_raises(cam):
    with pytest.raises(ValueError, match="corners visible"):
        estimate_extrinsics(SPEC, {0: _intr_truth()}, {0: View(np.zeros((3, 2)), np.arange(3, dtype=np.int32))})


def test_store_roundtrip_versions_and_resolution(tmp_path, cam, views):
    vs, imgs = views
    r0 = fit_intrinsics(SPEC, vs, SIZE, cam_id=0, seed=1)
    r1 = fit_intrinsics(SPEC, vs, SIZE, cam_id=1, seed=2)
    save_intrinsics(tmp_path, r0, SPEC, dict(list(imgs.items())[:2]))
    back, spec_back = load_intrinsics(tmp_path, 0)
    assert np.allclose(back.K_raw, r0.K_raw) and spec_back == SPEC and back.rms_heldout == pytest.approx(r0.rms_heldout)
    with pytest.raises(CalibrationError):
        load_intrinsics(tmp_path, 5)

    _poses, v = _two_cams(cam)
    ext = estimate_extrinsics(SPEC, {0: r0, 1: r1}, v)
    assert list_versions(tmp_path) == []
    v1 = save_calibration(tmp_path, SPEC, {0: r0, 1: r1}, ext, {0: imgs["v00.png"]} if "v00.png" in imgs else None)
    v2 = save_calibration(tmp_path, SPEC, {0: r0, 1: r1}, ext)
    assert (v1, v2) == (1, 2) and list_versions(tmp_path) == [1, 2] and active_version(tmp_path) == 2
    assert (tmp_path / "calib_v001.yaml").is_file() and (tmp_path / "calib_v001" / "manifest.json").is_file()

    cs = load_calibration(tmp_path)
    assert cs.version == 2 and cs.board == SPEC
    old = load_calibration(tmp_path, 1)
    assert old.version == 1
    rec = cs.cameras[0]
    c = rec.calibration
    assert c.calib_version == 2 and (c.width, c.height) == SIZE
    assert np.allclose(c.K, r0.K_undist) and np.allclose(c.dist, r0.dist)
    assert np.allclose(c.T_world_from_camera, ext.cameras[0].T_world_from_camera)

    und = rec.undistorter()
    frame = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    assert und.apply(frame).shape == frame.shape
    with pytest.raises(CalibrationError, match="1280x720"):
        und.apply(np.zeros((480, 640), np.uint8))
    with pytest.raises(CalibrationError):
        check_frame_size(c, (480, 640))
    check_frame_size(c, (SIZE[1], SIZE[0]))


def test_undistorted_image_matches_undistorted_K(cam):
    """Corners detected in the undistorted frame agree with projecting through the stored undistorted K."""
    r = _intr_truth()  # exact intrinsics: isolates the undistortion/K policy from fit error
    Ku = r.K_undist
    rng = np.random.default_rng(11)
    T = look_at([0.3, -0.5, 0.6], [0, 0, 0])
    img = cam.render(T, rng, noise_sigma=0.5)
    m1, m2 = cv2.initUndistortRectifyMap(r.K_raw, r.dist, None, Ku, SIZE, cv2.CV_32FC1)
    und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
    c, ids = detect_charuco(SPEC, und)
    from livefire.common.geometry import CameraCalibration
    cal = CameraCalibration(0, *SIZE, Ku, np.zeros(5), T, 1)
    uv, _, _ = project(cal, board_corners_world(SPEC)[ids])
    assert len(ids) >= 20 and np.median(np.linalg.norm(uv - c, axis=1)) < 0.4


def _projected_views(T_world_from_camera, n, sigma, seed):
    """n noisy detections (all board corners projected through the true model + Gaussian pixel noise)."""
    from livefire.calib.board import board_corners_board
    ids = np.arange(len(board_corners_board(SPEC)), dtype=np.int32)
    T_cb = np.linalg.inv(T_world_from_board(SPEC)) @ T_world_from_camera
    T_cb = np.linalg.inv(T_cb)  # camera_from_board
    rvec, _ = cv2.Rodrigues(T_cb[:3, :3])
    px, _ = cv2.projectPoints(board_corners_board(SPEC), rvec, T_cb[:3, 3], K_TRUE, DIST_TRUE)
    rng = np.random.default_rng(seed)
    return [View(px.reshape(-1, 2) + rng.normal(0, sigma, (len(ids), 2)), ids) for _ in range(n)]


def test_extrinsics_multiframe_average_beats_single_frame():
    T = look_at([0.5, -0.4, 0.6], [0.0, 0.0, 0.0])
    vs = _projected_views(T, 15, 1.5, seed=11)
    intr = {0: _intr_truth(0)}
    single = [estimate_extrinsics(SPEC, intr, {0: v}).cameras[0].T_world_from_camera for v in vs]
    err1 = np.median([np.linalg.norm(S[:3, 3] - T[:3, 3]) for S in single])
    res = estimate_extrinsics(SPEC, intr, {0: vs})
    c = res.cameras[0]
    err_avg = np.linalg.norm(c.T_world_from_camera[:3, 3] - T[:3, 3])
    assert c.n_frames == 15 and err_avg < 0.6 * err1
    assert _rot_deg(c.T_world_from_camera[:3, :3], T[:3, :3]) < np.median([_rot_deg(S[:3, :3], T[:3, :3]) for S in single])
    assert 0.0 < c.spread_mm < 20.0 and 0.0 < c.spread_deg < 2.0   # spread reported and consistent with the noise
    assert "pose spread" in res.summary()


def test_extrinsics_rejects_outlier_frame():
    T = look_at([0.5, -0.4, 0.6], [0.0, 0.0, 0.0])
    vs = _projected_views(T, 9, 0.3, seed=2)
    bad = View(vs[0].corners + np.random.default_rng(0).normal(0, 12.0, vs[0].corners.shape), vs[0].ids)
    res = estimate_extrinsics(SPEC, {0: _intr_truth(0)}, {0: [*vs, bad]})
    assert res.cameras[0].n_frames == 9 and any("rejected 1 of 10" in w for w in res.warnings)
    assert np.linalg.norm(res.cameras[0].T_world_from_camera[:3, 3] - T[:3, 3]) < 0.003


def test_extrinsics_camera_below_table_raises_clear_error():
    from livefire.calib.board import CalibrationError
    T = look_at([0.5, -0.4, -0.6], [0.0, 0.0, 0.0])          # camera under the table plane
    v = _projected_views(T, 1, 0.2, seed=1)[0]
    with pytest.raises(CalibrationError, match="wrong side"):
        estimate_extrinsics(SPEC, {0: _intr_truth(0)}, {0: v})


def test_pose_distance_unrelated_views_is_inf_and_never_still():
    from livefire.calib.intrinsics import pose_distance
    a = View(np.random.default_rng(0).uniform(0, 500, (12, 2)), np.arange(12, dtype=np.int32))
    b = View(a.corners.copy(), np.arange(100, 112, dtype=np.int32))         # disjoint ids: unrelated pose
    assert pose_distance(a, b, 1.0) == float("inf") and pose_distance(a, a, 1.0) == 0.0
    gate = CaptureGate(SIZE, min_corners=12, still_px=1.5, still_frames=2, min_pose_diff=0.05)
    out = [gate.update(v) for v in (a, b, a, b, a, b)]       # alternating unrelated poses are never "still"
    assert all(o is None for o in out) and not gate.captured


def test_capture_gate_and_coverage(views):
    vs, _ = views
    gate = CaptureGate(SIZE, min_corners=12, still_px=1.5, still_frames=3, min_pose_diff=0.05)
    got = [gate.update(vs[0]) for _ in range(6)]
    assert sum(g is not None for g in got) == 1                  # captured once while still, no duplicate
    assert gate.update(None) is None
    moving = [gate.update(v) for v in vs[1:6]]                   # always moving: nothing captured
    assert all(m is None for m in moving)
    cm = CoverageMap(SIZE)
    for v in vs:
        cm.add(v.corners)
    assert 0.2 < cm.covered_fraction <= 1.0 and cm.heatmap().shape == (SIZE[1], SIZE[0], 3)


def test_cli_end_to_end(tmp_path, cam, views, capsys):
    """board -> intrinsics --from-dir (2 cams) -> extrinsics --from-dir, via main(argv); versions increment."""
    from livefire.app.calib_cli import main
    _vs, imgs = views
    assert main(["board", "--out", str(tmp_path / "b.png")]) == 0
    assert "100 %" in capsys.readouterr().out and (tmp_path / "b.png").is_file()
    d = tmp_path / "imgs"
    d.mkdir()
    for name, im in imgs.items():
        cv2.imwrite(str(d / name), im)
    root = tmp_path / "calib"
    for c in (0, 1):
        assert main(["--root", str(root), "intrinsics", "--cam", str(c), "--from-dir", str(d)]) == 0
    assert "HELD-OUT" in capsys.readouterr().out
    poses, _ = _two_cams(cam)
    e = tmp_path / "ext"
    e.mkdir()
    rng = np.random.default_rng(5)
    for cid, T in poses.items():
        cv2.imwrite(str(e / f"cam{cid}.png"), cam.render(T, rng))
    assert main(["--root", str(root), "extrinsics", "--cams", "0,1", "--from-dir", str(e)]) == 0
    assert main(["--root", str(root), "extrinsics", "--cams", "0,1", "--from-dir", str(e)]) == 0
    assert list_versions(root) == [1, 2]
    cs = load_calibration(root)
    assert cs.cameras[0].calibration.T_world_from_camera[2, 3] > 0.3
    # a missing intrinsics store fails loudly (exit 1) instead of crashing
    assert main(["--root", str(tmp_path / "none"), "extrinsics", "--cams", "0", "--from-dir", str(e)]) == 1
