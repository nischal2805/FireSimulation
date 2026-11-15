"""P1-007: Gate 1 evaluation tools (M2 latency, M4 tape MAE + depth AbsRel/completeness, M6 alignment, placement study)
on synthetic data with known answers. No camera, GPU or display: the ``measure`` window layer is only exercised through
its pure parts (``MeasureSession``, ``render``, ``handle_key``)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from livefire.app.acq_bundle import AcqBundle, acquire_synthetic_bundle
from livefire.app.live import LATENCY_COLUMNS, LatencyRecorder
from livefire.app.measure_cli import MeasureSession, handle_key, layout_for, render
from livefire.app.measure_cli import main as measure_main
from livefire.common.config import Params, load_yaml
from livefire.common.eval_csv import read_csv, write_csv
from livefire.common.geometry import AABB, CameraCalibration, look_at, project
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

EVAL = Path(__file__).resolve().parents[1] / "eval"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"p1_eval_{name}", EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


latency, placement, alignment, geometry_mae = (_load(n) for n in ("latency", "placement", "alignment", "geometry_mae"))

SCENE = build_table_v1()
CAMS = load_cameras()
ROI = AABB(np.array([-0.8, -0.6, -0.05]), np.array([0.8, 0.6, 0.6]))
_RENDERS: dict[tuple, object] = {}


def _truth(cal: CameraCalibration):
    key = tuple(np.round(cal.T_world_from_camera, 6).ravel())
    if key not in _RENDERS:
        _RENDERS[key] = render_view(SCENE, cal)
    return _RENDERS[key]


def truth_bundle(cams: dict[int, CameraCalibration], label: str = "synthetic truth") -> AcqBundle:
    """# MOCK: bundle whose frozen depth is the renderer's exact ground truth (hole = miss), not an acquisition result."""
    tv = {c: _truth(cal) for c, cal in cams.items()}
    return AcqBundle(dict(cams), {c: t.depth.copy() for c, t in tv.items()}, {c: ~t.valid for c, t in tv.items()},
                     {c: t.rgb.copy() for c, t in tv.items()}, ROI, 0.01, None, label)  # fmt: skip


def orbit_cam(cam_id: int, az_deg: float, elev_deg: float = 45.0, radius: float = 1.2) -> CameraCalibration:
    """Camera on a circle around the table centre looking at (0, 0, 0.04), intrinsics of the synthetic rig."""
    az, el = np.radians(az_deg), np.radians(elev_deg)
    pos = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    c0 = CAMS[0]
    return CameraCalibration(cam_id, c0.width, c0.height, c0.K, c0.dist, look_at(pos, [0, 0, 0.04]), c0.calib_version)


def tight_params():
    """Placement thresholds with a 8 mm visibility depth tolerance (default 3 cm hides occluders closer than that)."""
    raw = dict(load_yaml("placement")["placement"])
    raw.pop("region_xy_m")
    raw["visible_depth_tol_m"] = {**raw["visible_depth_tol_m"], "value": 0.008}
    return Params(raw)


def raycast_visible(cal: CameraCalibration, pts: np.ndarray) -> np.ndarray:
    """Independent visibility of table samples ``pts`` (N,3): inside the image and no scene object (table excluded)
    between the point and the camera centre, by analytic ray casting from the point towards the camera."""
    uv, _, front = project(cal, pts)
    inside = front & (uv[:, 0] >= -0.5) & (uv[:, 0] < cal.width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < cal.height - 0.5)
    o = cal.center_world
    vis = inside.copy()
    for i in np.flatnonzero(inside):
        start = pts[i] + np.array([0.0, 0.0, 1e-4])
        d = o - start
        dist = float(np.linalg.norm(d))
        if any(np.all(start > np.asarray(o.lo)) and np.all(start < np.asarray(o.hi)) for o in SCENE.objects[1:3]):  # wood, metal boxes
            vis[i] = False  # the sample lies under an object (inside its footprint)
            continue
        for obj in SCENE.objects[1:]:  # wood, metal, paper, cloth
            t, _ = obj.intersect(start, (d / dist)[None, :])
            if t[0] < dist:
                vis[i] = False
                break
    return vis


REGION = (-0.6, -0.35, 0.6, 0.35)  # inside the table_v1 top (|x| < 0.8, |y| < 0.5)


# ------------------------------------------------------------------------------------------------ latency (M2)
def _csv_rows(path: Path, rows: list[dict]) -> Path:
    return write_csv(path, LATENCY_COLUMNS, rows)


def _row(frame, cam, state, c2d, t_present_ns, frame_ms=1000 / 30, dropped=0, lag=0.0, age=20.0):
    return {"frame": frame, "cam": cam, "seq": frame, "state": state, "t_arrive_ns": t_present_ns - int(c2d * 1e6),
            "t_use_ns": t_present_ns - int(c2d * 1e6) + int(age * 1e6), "t_present_ns": t_present_ns, "frame_age_ms": age,
            "capture_to_display_ms": c2d, "composite_ms": 8.0, "frame_ms": frame_ms, "dropped_frame": dropped,
            "sim_lag_s": lag, "fps": 30.0, "cams_shown": 2}  # fmt: skip


def _synthetic_latency_csv(path: Path) -> Path:
    """Known answers: cam0 c2d = 50 + frame % 11 (50..60, p50 55), cam1 = 105 + frame % 21 (105..125, p50 115, p95 124).
    Frames 0-29 warm-up (c2d 999), frames 30-39 READY (7777), 40-239 SIMULATING at exactly 30 fps; 4 dropped frames."""
    rows, t0 = [], 1_000_000_000
    for f in range(240):
        tp = t0 + round(f * 1e9 / 30)
        for cam in (0, 1):
            if f < 30:
                st, c2d = "SIMULATING", 999.0
            elif f < 40:
                st, c2d = "READY", 7777.0
            else:
                st, c2d = "SIMULATING", 50.0 + f % 11 if cam == 0 else 105.0 + f % 21
            rows.append(_row(f, cam, st, c2d, tp, frame_ms=0.0 if f == 0 else 1000 / 30, dropped=int(f in (50, 90, 130, 170)),
                             lag=0.001 * f))  # fmt: skip
    return _csv_rows(path, rows)


def test_latency_statistics_known_answers(tmp_path: Path) -> None:
    rows = latency.load_latency_csv(_synthetic_latency_csv(tmp_path / "latency.csv"))
    out = {r["scope"]: r for r in latency.m2_statistics(rows)}  # defaults: states SIMULATING/PAUSED, skip 30 frames
    assert set(out) == {"all", "cam0", "cam1"}
    c0, c1, al = out["cam0"], out["cam1"], out["all"]
    assert c0["n_rows"] == 200 and c0["n_frames"] == 200  # frames 40..239 only: warm-up and READY rows are excluded
    f = np.arange(40, 240)
    assert c0["c2d_p50_ms"] == pytest.approx(np.median(50 + f % 11)) == 55.0
    assert c0["c2d_max_ms"] == 60.0 and c0["meets_target_p50"] is True
    assert c1["c2d_p50_ms"] == pytest.approx(np.median(105 + f % 21)) and 113 <= c1["c2d_p50_ms"] <= 117
    assert c1["c2d_max_ms"] == 125.0 and c1["c2d_p95_ms"] > 120.0 and c1["meets_target_p50"] is False
    assert al["n_rows"] == 400 and al["n_frames"] == 200  # frame-level numbers count each frame once, not once per camera
    assert al["fps_delivered"] == pytest.approx(30.0, rel=1e-3)
    assert al["frame_p95_ms"] == pytest.approx(1000 / 30)
    assert al["dropped_rate"] == pytest.approx(4 / 200)
    assert al["sim_lag_max_s"] == pytest.approx(0.239) and al["sim_lag_end_s"] == pytest.approx(0.239)


def test_latency_matches_live_recorder_stats(tmp_path: Path) -> None:
    """Cross-check against ``LatencyRecorder.stats()`` (what the HUD shows) on a CSV written by the real recorder."""
    rec = LatencyRecorder(tmp_path / "latency.csv", window=120)
    rng = np.random.default_rng(0)
    t = 5_000_000_000
    for f in range(50):
        t += 33_000_000
        dropped = rec.add_frame(t, 33.0)
        for cam in (0, 1):
            rec.add_row(_row(f, cam, "SIMULATING", float(rng.uniform(40, 120)), t, dropped=int(dropped)))
    rec.close()
    s = rec.stats()
    me = {r["scope"]: r for r in latency.m2_statistics(latency.load_latency_csv(tmp_path / "latency.csv"), skip_frames=0)}["all"]
    assert me["c2d_p50_ms"] == pytest.approx(s["c2d_p50_ms"]) and me["c2d_p95_ms"] == pytest.approx(s["c2d_p95_ms"])
    assert me["c2d_p99_ms"] == pytest.approx(s["c2d_p99_ms"])


def test_latency_cli_glass_to_glass_stays_separate(tmp_path: Path, capsys) -> None:
    src = _synthetic_latency_csv(tmp_path / "latency.csv")
    rc = latency.main([str(src), "--label", "unit", "--out-dir", str(tmp_path), "--glass", "0=100,120,110", "--glass", "1=140"])
    assert rc == 0
    rows = {r["scope"]: r for r in read_csv(tmp_path / "latency_unit.csv")}
    assert float(rows["cam0"]["g2g_offset_ms"]) == 110.0 and rows["cam0"]["g2g_n"] == "3"
    assert float(rows["cam0"]["c2d_p50_ms"]) == 55.0  # M2 itself does not include the offset
    assert float(rows["cam0"]["c2d_p50_plus_offset_ms"]) == 165.0 and rows["all"]["g2g_offset_ms"] == ""
    assert "NOT MET" in capsys.readouterr().out  # cam1 median over 100 ms
    assert latency.main([str(tmp_path / "nope.csv"), "--out-dir", str(tmp_path)]) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert latency.main([str(bad), "--out-dir", str(tmp_path)]) == 2
    with pytest.raises(ValueError):
        latency.parse_glass(["0:100"])


def test_latency_no_rows_after_filter_raises(tmp_path: Path) -> None:
    rows = latency.load_latency_csv(_synthetic_latency_csv(tmp_path / "latency.csv"))
    with pytest.raises(ValueError, match="no rows"):
        latency.m2_statistics(rows, states=("ERROR",))


# ------------------------------------------------------------------------------------------------ placement
def test_placement_good_rig_passes_and_angle_is_analytic() -> None:
    cams = {0: orbit_cam(0, 0.0), 1: orbit_cam(1, 90.0)}
    rep = placement.placement_study(truth_bundle(cams), region_xy=REGION, params=tight_params())
    # angle at the table centre between two cameras at elevation e, azimuth difference d: acos(sin^2 e + cos^2 e cos d) = 60 deg
    assert rep.angles_deg[(0, 1)] == pytest.approx(60.0, abs=0.5)  # centre is (0,0,0), cameras look at z = 0.04
    assert rep.ok, rep.guidance
    assert all(m["valid_coverage"] == 1.0 for m in rep.per_cam.values())
    pts = placement.table_samples(REGION, 0.02)
    vis = [raycast_visible(cams[c], pts) for c in (0, 1)]  # independent of the depth maps
    assert rep.per_cam[0]["visible_fraction"] == pytest.approx(vis[0].mean(), abs=0.01)
    assert rep.per_cam[1]["visible_fraction"] == pytest.approx(vis[1].mean(), abs=0.01)
    assert rep.shared_fraction == pytest.approx((vis[0] & vis[1]).mean(), abs=0.01) and 0.5 < rep.shared_fraction < 0.95
    assert rep.per_cam[0]["distance_m"] == pytest.approx(1.2, abs=1e-6)
    rep.raise_if_failed()
    assert not rep.warnings  # 1.2 m, 60 deg: inside the recommended bands


def test_placement_close_cameras_fail_with_guidance() -> None:
    cams = {0: orbit_cam(0, 0.0), 1: orbit_cam(1, 20.0)}
    rep = placement.placement_study(truth_bundle(cams), region_xy=REGION)
    expected = np.degrees(np.arccos(0.5 + 0.5 * np.cos(np.radians(20.0))))  # 14.2 deg
    assert rep.angles_deg[(0, 1)] == pytest.approx(expected, abs=0.05)
    assert not rep.ok and any("apart" in f for f in rep.failures)
    with pytest.raises(placement.PlacementError, match="sideways"):
        rep.raise_if_failed()


def test_placement_wide_angle_fails() -> None:
    rep = placement.placement_study(truth_bundle({0: orbit_cam(0, 0.0, 20.0), 1: orbit_cam(1, 170.0, 20.0)}), region_xy=REGION)
    expected = np.degrees(np.arccos(np.sin(np.radians(20)) ** 2 + np.cos(np.radians(20)) ** 2 * np.cos(np.radians(170))))
    assert rep.angles_deg[(0, 1)] == pytest.approx(expected, abs=0.05) and rep.angles_deg[(0, 1)] > 100.0 and not rep.ok and any("closer together" in g for g in rep.guidance)


def test_placement_camera_looking_away_fails_shared_and_visible() -> None:
    away = replace(orbit_cam(1, 90.0), T_world_from_camera=look_at(orbit_cam(1, 90.0).center_world, [0.0, 3.0, 0.6]))
    rep = placement.placement_study(truth_bundle({0: orbit_cam(0, 0.0), 1: away}), region_xy=REGION)
    assert rep.per_cam[1]["in_frame"] < 0.05 and rep.shared_fraction < 0.05
    assert not rep.ok and any("cam1: sees only" in f for f in rep.failures) and any("seen by all cameras" in f for f in rep.failures)


def test_placement_valid_coverage_matches_hole_fraction() -> None:
    cams = {0: orbit_cam(0, 0.0), 1: orbit_cam(1, 90.0)}
    b = truth_bundle(cams)
    hole = b.view_hole[1].copy()
    hole[:, : cams[1].width // 2] = True  # left half of cam1 has no depth
    depth = b.view_depth[1].copy()
    depth[hole] = 0.0
    b = replace(b, view_hole={**b.view_hole, 1: hole}, view_depth={**b.view_depth, 1: depth})
    pts = placement.table_samples(REGION, 0.02)
    uv, _, front = project(cams[1], pts)
    inside = front & (uv[:, 0] >= -0.5) & (uv[:, 0] < cams[1].width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < cams[1].height - 0.5)
    expected = float((inside & (uv[:, 0] >= cams[1].width / 2 - 0.5)).sum() / inside.sum())
    rep = placement.placement_study(b, region_xy=REGION)
    assert rep.per_cam[1]["valid_coverage"] == pytest.approx(expected, abs=0.01) and 0.0 < expected < 1.0
    assert rep.per_cam[0]["valid_coverage"] == 1.0
    assert not rep.ok and any(f.startswith("cam1: only") and "valid depth" in f for f in rep.failures)


def test_placement_single_camera_allowed_with_warning() -> None:
    rep = placement.placement_study(truth_bundle({0: orbit_cam(0, 0.0)}), region_xy=REGION)
    assert rep.ok and rep.shared_fraction is None and rep.angles_deg == {}
    assert any("single camera" in w for w in rep.warnings)


def test_placement_cli_exit_codes_and_csv(tmp_path: Path, capsys) -> None:
    good = truth_bundle({0: orbit_cam(0, 0.0), 1: orbit_cam(1, 90.0)}, "unit good").save(tmp_path / "good.npz", write_pngs=False)
    bad = truth_bundle({0: orbit_cam(0, 0.0), 1: orbit_cam(1, 15.0)}, "unit bad").save(tmp_path / "bad.npz", write_pngs=False)
    out = tmp_path / "res"
    assert placement.main(["--bundle", str(good), "--region=" + ",".join(map(str, REGION)), "--out-dir", str(out)]) == 0
    assert placement.main(["--bundle", str(bad), "--region=" + ",".join(map(str, REGION)), "--out-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert "PLACEMENT FAILED" in err and "apart" in err
    rows = read_csv(out / "placement_unit_bad.csv")
    assert {r["metric"] for r in rows} >= {"camera_angle", "shared_fraction", "valid_coverage", "visible_fraction"}
    assert any(r["status"] == "FAIL" for r in rows)
    assert placement.main(["--bundle", str(good), "--region", "1,1,0,0"]) == 2
    assert placement.main(["--bundle", str(good), "--region=a,b"]) == 2
    assert placement.main(["--bundle", str(tmp_path / "missing.npz"), "--out-dir", str(out)]) == 2
    assert "placement study failed" in capsys.readouterr().err
    assert placement.main(["--bundle", str(good), "--label", "sceneB", "--out-dir", str(out)]) in (0, 1)
    assert (out / "placement_sceneB.csv").is_file() and (out / "placement_unit_good.csv").is_file()


# ------------------------------------------------------------------------------------------------ bundle
def test_bundle_roundtrip_and_snapshot(tmp_path: Path) -> None:
    b = truth_bundle(CAMS, "roundtrip")
    p = b.save(tmp_path / "a" / "acquisition.npz")
    assert (tmp_path / "a" / "view_cam0.png").is_file() and (tmp_path / "a" / "depth_cam1.png").is_file()
    r = AcqBundle.load(p)
    assert r.label == "roundtrip" and sorted(r.cameras) == [0, 1] and r.board is None
    for c in (0, 1):
        assert np.array_equal(r.view_depth[c], b.view_depth[c]) and np.array_equal(r.view_hole[c], b.view_hole[c])
        assert np.array_equal(r.rgb[c], b.rgb[c]) and np.allclose(r.cameras[c].T_world_from_camera, CAMS[c].T_world_from_camera)
    snap = r.snapshot()
    assert snap.pos.shape == (0, 3) and set(snap.view_depth) == {0, 1} and not snap.view_depth[0].flags.writeable


# ------------------------------------------------------------------------------------------------ M4 measure
WOOD_TOP = [np.array([0.15, 0.10, 0.08]), np.array([0.25, 0.10, 0.08])]  # on the wood block top: 10 cm apart
TABLE_PTS = [np.array([-0.5, -0.3, 0.0]), np.array([0.5, -0.3, 0.0])]  # on the table top: 100 cm apart


def _pix(cam: CameraCalibration, p: np.ndarray) -> tuple[float, float]:
    uv, _, _ = project(cam, p)
    return float(uv[0, 0]), float(uv[0, 1])


def test_measure_pairs_reconstruct_known_distances() -> None:
    s = MeasureSession(truth_bundle(CAMS))
    for a, b, truth_cm in [(*WOOD_TOP, 10.0), (*TABLE_PTS, 100.0)]:
        assert s.click(0, *_pix(CAMS[0], a)) and s.awaiting_distance is False
        assert s.click(0, *_pix(CAMS[0], b)) and s.awaiting_distance
        pr = s.submit(truth_cm)
        assert pr.d_recon_cm == pytest.approx(truth_cm, abs=0.5)  # centimetre-level on exact depth (pixel quantisation only)
        assert abs(pr.error_cm) < 0.5
    # a pair with one point seen by each camera: world points are camera independent
    assert s.click(0, *_pix(CAMS[0], TABLE_PTS[0])) and s.click(1, *_pix(CAMS[1], TABLE_PTS[1]))
    assert s.submit(100.0).d_recon_cm == pytest.approx(100.0, abs=1.0)


def test_measure_mae_known_offsets_and_csv(tmp_path: Path) -> None:
    s = MeasureSession(truth_bundle(CAMS))
    offsets = [1.0, -2.0, 0.5, 3.5]  # tape = reconstructed - offset  ->  error_cm = +offset exactly
    pts = [WOOD_TOP, TABLE_PTS, [TABLE_PTS[0], WOOD_TOP[0]], [TABLE_PTS[1], WOOD_TOP[1]]]
    for (a, b), off in zip(pts, offsets):
        assert s.click(0, *_pix(CAMS[0], a)) and s.click(0, *_pix(CAMS[0], b))
        recon = float(np.linalg.norm(s.pending[0].world - s.pending[1].world)) * 100
        s.submit(recon - off)
    rep = s.report()
    assert rep["n"] == 4 and rep["mae_cm"] == pytest.approx(np.mean(np.abs(offsets)))
    assert rep["bias_cm"] == pytest.approx(np.mean(offsets)) and rep["max_abs_cm"] == pytest.approx(3.5)
    path = s.save(tmp_path / "m" / "pairs.csv")
    err, tape = geometry_mae.read_pairs(path)
    assert np.allclose(err, offsets)
    full = geometry_mae.mae_report(err, tape)
    assert full["mae_cm"] == pytest.approx(rep["mae_cm"]) and full["rmse_cm"] == pytest.approx(np.sqrt(np.mean(np.square(offsets))))
    assert read_csv(path)[0].keys() >= {"id", "cam_a", "u_a", "d_recon_cm", "d_tape_cm", "error_cm"}


def test_measure_typing_hole_rejection_undo() -> None:
    b = truth_bundle(CAMS)
    hole = b.view_hole[0].copy()
    hole[:100, :100] = True
    s = MeasureSession(replace(b, view_hole={**b.view_hole, 0: hole}))
    assert not s.click(0, 50, 50) and "hole" in s.message and not s.pending
    assert not s.click(0, -5, 20) and "outside" in s.message
    assert not s.click(7, 300, 300) and "no frozen depth" in s.message  # unknown camera
    s.type_char("5")  # ignored: no pair pending
    assert s.typed == ""
    assert s.click(0, *_pix(CAMS[0], WOOD_TOP[0])) and s.click(0, *_pix(CAMS[0], WOOD_TOP[1]))
    assert not s.click(0, 400, 400) and "tape distance" in s.message  # pair waits for its distance
    assert s.enter() is None and "bad distance" in s.message  # nothing typed
    for ch in "1x2..5":
        s.type_char(ch)
    assert s.typed == "12.5"  # letter and second dot ignored
    s.backspace()
    s.backspace()
    assert s.typed == "12" and s.enter().tape_cm == 12.0
    s.type_char("0")
    with pytest.raises(ValueError):
        s.submit(5.0)  # nothing pending
    assert s.click(0, 400, 400) and s.click(0, 420, 420)
    with pytest.raises(ValueError):
        s.submit(0.0)
    s.undo()  # cancels the open pair first
    assert s.pending == [] and len(s.pairs) == 1
    s.undo()
    assert s.pairs == [] and s.report is not None
    with pytest.raises(ValueError):
        s.report()


def test_measure_keys_and_render() -> None:
    s = MeasureSession(truth_bundle(CAMS))
    mixed = {0: CAMS[0], 1: replace(CAMS[1], width=640, height=360)}
    with pytest.raises(ValueError, match="same resolution"):
        layout_for(replace(s.bundle, cameras=mixed), 1400, 600)
    lay = layout_for(s.bundle, 1400, 600)
    before = render(s, lay)
    assert before.shape == (lay.pane_h + 70, lay.pane_w * 2, 3) and before.dtype == np.uint8
    u, v = _pix(CAMS[0], WOOD_TOP[0])
    assert s.click(0, u, v)
    x, y = lay.from_view(0, u, v)
    after = render(s, lay)
    assert np.abs(after[int(y) - 8 : int(y) + 9, int(x) - 8 : int(x) + 9].astype(int) - before[int(y) - 8 : int(y) + 9, int(x) - 8 : int(x) + 9]).sum() > 0
    assert lay.to_view(x, y)[0] == 0
    assert s.click(1, *_pix(CAMS[1], WOOD_TOP[1]))
    for ch in "10":
        assert handle_key(s, ord(ch)) is None
    assert s.typed == "10" and handle_key(s, 8) is None and s.typed == "1"
    assert handle_key(s, 13) is None and len(s.pairs) == 1 and s.pairs[0].tape_cm == 1.0
    assert handle_key(s, ord("u")) is None and s.pairs == []
    assert handle_key(s, -1) is None and handle_key(s, ord("s")) == "save" and handle_key(s, ord("q")) == "quit"
    assert s.click(0, *_pix(CAMS[0], TABLE_PTS[0])) and s.click(0, *_pix(CAMS[0], TABLE_PTS[1]))
    assert handle_key(s, 27) is None and not s.pending  # Esc while typing cancels the pair, does not quit
    assert handle_key(s, 27) == "quit"
    nofr = MeasureSession(replace(truth_bundle(CAMS), rgb={}))
    with pytest.raises(ValueError, match="no camera frames"):
        render(nofr, lay)


def test_measure_cli_headless_from_csv(tmp_path: Path, capsys) -> None:
    bundle = truth_bundle(CAMS).save(tmp_path / "acq" / "acquisition.npz", write_pngs=False)
    a, b = (_pix(CAMS[0], p) for p in TABLE_PTS)
    src = tmp_path / "pixels.csv"
    src.write_text("cam_a,u_a,v_a,cam_b,u_b,v_b,tape_cm\n" + f"0,{a[0]},{a[1]},0,{b[0]},{b[1]},101.0\n")
    out = tmp_path / "pairs.csv"
    assert measure_main(["--bundle", str(bundle), "--from-csv", str(src), "--pairs-out", str(out)]) == 0
    assert "M4: n=1" in capsys.readouterr().out
    err, _ = geometry_mae.read_pairs(out)
    assert err[0] == pytest.approx(-1.0, abs=0.5)  # reconstructed ~100 cm, tape 101
    src.write_text("cam_a,u_a,v_a,cam_b,u_b,v_b,tape_cm\n0,5,5,0,6,6,10\n0,0,0,0,1,1,5\n")
    hole_bundle = replace(AcqBundle.load(bundle), view_hole={0: np.ones((720, 1280), bool), 1: np.ones((720, 1280), bool)})
    hole_bundle.save(tmp_path / "holes.npz", write_pngs=False)
    assert measure_main(["--bundle", str(tmp_path / "holes.npz"), "--from-csv", str(src), "--pairs-out", str(out)]) == 2


# ------------------------------------------------------------------------------------------------ M4 depth metrics
def test_mae_report_known_answer() -> None:
    err = np.array([1.0, -1.0, 2.0, -2.0])
    r = geometry_mae.mae_report(err, np.array([10.0, 20.0, 40.0, 50.0]))
    assert r["n"] == 4 and r["mae_cm"] == 1.5 and r["bias_cm"] == 0.0 and r["max_abs_cm"] == 2.0
    assert r["rmse_cm"] == pytest.approx(np.sqrt(2.5)) and r["median_abs_cm"] == 1.5
    assert r["mean_rel_err_pct"] == pytest.approx(100 * np.mean([0.1, 0.05, 0.05, 0.04]))
    with pytest.raises(ValueError):
        geometry_mae.mae_report(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        geometry_mae.mae_report(np.array([1.0]), np.array([0.0]))


def test_depth_metrics_absrel_and_completeness() -> None:
    gt = np.full((10, 10), 2.0, np.float32)
    valid = np.ones((10, 10), bool)
    depth = gt * 1.1  # 10 % too far everywhere
    hole = np.zeros((10, 10), bool)
    hole[:2] = True  # 20 % missing
    m = geometry_mae.depth_metrics(np.where(hole, 0, depth), hole, gt, valid)
    assert m["absrel"] == pytest.approx(0.1, rel=1e-5) and m["mae_m"] == pytest.approx(0.2, rel=1e-5)
    assert m["completeness"] == pytest.approx(0.8) and m["n_gt"] == 100
    mask = np.zeros((10, 10), bool)
    mask[:5] = True  # ROI: top half, of which 2 of 5 rows are holes
    assert geometry_mae.depth_metrics(np.where(hole, 0, depth), hole, gt, valid, mask)["completeness"] == pytest.approx(0.6)
    nogt = geometry_mae.depth_metrics(depth, hole, gt, np.zeros((10, 10), bool))
    assert nogt["n_gt"] == 0 and np.isnan(nogt["absrel"])


def test_synthetic_depth_metrics_on_truth_and_scaled_depth() -> None:
    b = truth_bundle(CAMS)
    truth = {c: _truth(cal) for c, cal in CAMS.items()}
    for m in geometry_mae.synthetic_depth_metrics(b, truth).values():
        assert m["mae_m"] == 0.0 and m["absrel"] == 0.0 and m["completeness"] == 1.0 and m["n_gt"] > 10_000
    scaled = replace(b, view_depth={c: d * 1.05 for c, d in b.view_depth.items()})
    for m in geometry_mae.synthetic_depth_metrics(scaled, truth).values():
        assert m["absrel"] == pytest.approx(0.05, rel=1e-3)


# ------------------------------------------------------------------------------------------------ M6 alignment
def _expected_ray_error(cal: CameraCalibration, p_world: np.ndarray, dz: float) -> float:
    """A Z-depth offset ``dz`` moves the back-projected point along the pixel ray by ``dz * |P_c| / P_c.z``."""
    pc = (np.linalg.inv(cal.T_world_from_camera) @ np.append(p_world, 1.0))[:3]
    return dz * float(np.linalg.norm(pc) / pc[2])


def test_alignment_known_depth_offsets() -> None:
    for offset in (0.0, 0.02, 0.05):
        b = alignment.selftest_bundle(offset)
        lms = alignment.board_landmarks(b)
        assert {lm.cam for lm in lms} == {0, 1} and len(lms) >= 30
        rows = alignment.landmark_errors(b, lms)
        assert all(r["status"] == "ok" for r in rows)
        err = np.array([r["err_m"] for r in rows])
        expected = np.array([_expected_ray_error(b.cameras[lm.cam], lm.world, offset) for lm in lms])
        assert np.abs(err - expected).max() < 0.004, (offset, np.abs(err - expected).max())  # corner detection + 3x3 depth median
        summ = {(s["scope"], s["metric"]): s["value"] for s in alignment.summarize(rows)}
        assert summ["all", "err_median_cm"] == pytest.approx(np.median(expected) * 100, abs=0.4)
        assert summ["all", "reconstructed_fraction"] == 1.0
        assert summ["all", "meets_m6"] == (1.0 if np.median(expected) < 0.03 else 0.0)


def test_alignment_holes_and_csv_landmarks(tmp_path: Path) -> None:
    b = alignment.selftest_bundle(0.0)
    cal = b.cameras[0]
    p_ok, p_hole = np.array([0.05, 0.03, 0.0]), np.array([-0.1, -0.02, 0.0])
    (u1, v1), (u2, v2) = _pix(cal, p_ok), _pix(cal, p_hole)
    hole = b.view_hole[0].copy()
    hole[int(v2) - 4 : int(v2) + 5, int(u2) - 4 : int(u2) + 5] = True
    b = replace(b, view_hole={**b.view_hole, 0: hole})
    f = tmp_path / "landmarks.csv"
    f.write_text("name,x,y,z,cam,u,v\n" + f"ok,{p_ok[0]},{p_ok[1]},0,0,{u1},{v1}\n" + f"hole,{p_hole[0]},{p_hole[1]},0,0,{u2},{v2}\n"
                 + f"elevated,{p_ok[0]},{p_ok[1]},0.05,0,{u1},{v1}\n")  # claims a point 5 cm above the plane at the same pixel
    rows = alignment.landmark_errors(b, alignment.read_landmarks(f))
    assert [r["status"] for r in rows] == ["ok", "hole", "ok"]
    assert rows[0]["err_m"] < 0.003 and np.isnan(rows[1]["err_m"])
    assert rows[2]["err_m"] == pytest.approx(0.05, abs=0.003) and rows[2]["err_z_m"] == pytest.approx(-0.05, abs=0.003)
    s = {(r["scope"], r["metric"]): r["value"] for r in alignment.summarize(rows)}
    assert s["all", "reconstructed_fraction"] == pytest.approx(2 / 3) and s["all", "landmarks_total"] == 3
    bad = tmp_path / "bad.csv"
    bad.write_text("name,x\na,1\n")
    with pytest.raises(ValueError, match="needs columns"):
        alignment.read_landmarks(bad)


def test_alignment_board_missing_and_no_frames() -> None:
    b = alignment.selftest_bundle(0.0)
    blank = replace(b, rgb={c: np.full_like(v, 128) for c, v in b.rgb.items()})
    assert alignment.board_landmarks(blank) == []
    with pytest.raises(ValueError, match="no camera frames"):
        alignment.board_landmarks(replace(b, rgb={}))
    with pytest.raises(ValueError, match="board spec"):
        alignment.board_landmarks(replace(b, board=None))


def test_alignment_cli_selftest(tmp_path: Path, capsys) -> None:
    assert alignment.main(["--selftest", "--offset-m", "0.04", "--out-dir", str(tmp_path), "--label", "st"]) == 0
    rows = read_csv(tmp_path / "alignment_st.csv")
    summ = {(r["scope"], r["metric"]): r["value"] for r in rows}
    assert 3.5 < float(summ["all", "err_median_cm"]) < 6.5  # 4 cm depth error along ~1.0-1.2x rays
    assert summ["all", "meets_m6"] == ""  # a self-test carries no verdict
    assert all("SYNTHETIC SELFTEST" in r["note"] and "selftest board" in r["label"] for r in rows)
    assert len(read_csv(tmp_path / "alignment_st_landmarks.csv")) >= 30
    assert "NOT MET" in capsys.readouterr().out
    assert alignment.main(["--selftest", "--offset-m", "0.0", "--out-dir", str(tmp_path), "--label", "st0"]) == 0
    out = capsys.readouterr().out
    assert "MET" in out and "NOT MET" not in out and "SYNTHETIC SELFTEST" in out and "not a fully independent" in out


# ------------------------------------------------------------------------------------------------ acquisition end to end
@pytest.fixture(scope="module")
def acquired() -> AcqBundle:
    return acquire_synthetic_bundle("oracle", grid=32, seconds=0.3, max_frames=6)


def test_acquired_synthetic_bundle_through_all_tools(acquired: AcqBundle, tmp_path: Path) -> None:
    """Real acquisition path (oracle depth, CPU): depth metrics are tight, placement of the synthetic rig is judged."""
    assert "oracle" in acquired.label and sorted(acquired.rgb) == [0, 1] and acquired.board is None
    truth = {c: _truth(cal) for c, cal in acquired.cameras.items()}
    for m in geometry_mae.synthetic_depth_metrics(acquired, truth).values():
        assert m["completeness"] > 0.95 and m["mae_m"] < 0.02 and m["absrel"] < 0.02, m
    path = acquired.save(tmp_path / "acquisition.npz", write_pngs=False)
    rc = placement.main(["--bundle", str(path), "--region=" + ",".join(map(str, REGION)), "--out-dir", str(tmp_path)])
    rows = {(r["metric"], r["scope"]): r for r in read_csv(next(tmp_path.glob("placement_*.csv")))}
    ang = float(rows["camera_angle", "cam0-cam1"]["value"])
    c0, c1 = acquired.cameras[0].center_world, acquired.cameras[1].center_world
    expected = np.degrees(np.arccos(c0 @ c1 / (np.linalg.norm(c0) * np.linalg.norm(c1))))  # table centre = origin
    assert ang == pytest.approx(expected, abs=0.01) and 40 < ang < 100
    assert rc == 0 and float(rows["shared_fraction", "all"]["value"]) > 0.5
    # a click through the acquired snapshot lands on the wood block top
    s = MeasureSession(acquired)
    assert s.click(0, *_pix(acquired.cameras[0], WOOD_TOP[0]))
    assert np.linalg.norm(s.pending[0].world - WOOD_TOP[0]) < 0.02


def test_geometry_mae_cli_pairs(tmp_path: Path, capsys) -> None:
    s = MeasureSession(truth_bundle(CAMS))
    for (a, b), tape in [(WOOD_TOP, 10.5), (TABLE_PTS, 99.0)]:
        s.click(0, *_pix(CAMS[0], a))
        s.click(0, *_pix(CAMS[0], b))
        s.submit(tape)
    pairs = s.save(tmp_path / "run1" / "pairs.csv")
    assert geometry_mae.main(["--pairs", str(pairs), "--out-dir", str(tmp_path), "--label", "unit"]) == 0
    rows = {r["metric"]: r for r in read_csv(tmp_path / "geometry_mae_unit.csv")}
    assert float(rows["mae_cm"]["value"]) == pytest.approx(np.mean([abs(10.0 - 10.5), abs(100.0 - 99.0)]), abs=0.5)
    assert rows["mae_cm"]["n"] == "2" and rows["mae_cm"]["unit"] == "cm"
    assert "WARNING: only 2 pairs" in capsys.readouterr().out
    assert geometry_mae.main(["--pairs", str(tmp_path / "missing.csv"), "--out-dir", str(tmp_path)]) == 2
