"""M6 fusion alignment (P1-007, PRD 11.1 M6): independent landmarks reconstructed vs known, median target under 3 cm.

A landmark is a point with a **known world position** that is visible at a **known pixel** of a camera. The
reconstructed position is the back-projection of that pixel through the *frozen* acquisition depth (the same path a
click uses, ``app.click.resolve_click``: depth of the fused geometry along the landmark's pixel ray, 3x3 neighbourhood
median, never borrowed across holes). Error = ``|reconstructed - known|`` (3-D, metres) plus its Z component.

Two landmark sources:

* **board** (default): the ChArUco corners of the printed board visible in the acquisition frames (detected in the
  undistorted frame, the pixel space of the frozen depth), known world positions from the calibration board spec
  (``calib.board.board_corners_world``). The board must **not have moved** since ``calib extrinsics``.
* **--landmarks landmarks.csv**: your own points (e.g. AprilTag centres, ruler marks) with columns
  ``name,x,y,z,cam,u,v`` (world metres; ``u,v`` undistorted pixel in the saved ``view_cam<c>.png`` of the bundle).
  Use this for landmarks that are *not* on the table plane.

Limitation (printed in the CSV note): depth is anchored to the table plane (P1R3) over the board footprint, so board
corners lying on that plane are not fully independent of the reconstruction. They still catch calibration/extrinsics
errors and depth-scale problems; add elevated landmarks via ``--landmarks`` for an independent check.

Output: ``eval/results/alignment_<label>.csv`` (summary rows) and ``alignment_<label>_landmarks.csv`` (one row per landmark).
``--selftest [--offset-m D]`` runs the whole pipeline on a synthetic board with a known depth offset D (default 0.02 m)
and prints the expected error; it needs no cameras.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from livefire.app.acq_bundle import AcqBundle, add_source_args, resolve_bundle
from livefire.app.click import resolve_click
from livefire.calib.board import BoardSpec, board_corners_world, detect_charuco
from livefire.common.eval_csv import percentile, read_csv, write_csv

TARGET_M = 0.03
MIN_BOARD_CORNERS = 8
SUMMARY_FIELDS = ("metric", "scope", "value", "unit", "n", "label", "note")
LANDMARK_FIELDS = ("name", "cam", "u", "v", "x_known", "y_known", "z_known", "x_recon", "y_recon", "z_recon",
                   "err_m", "err_z_m", "status")  # fmt: skip
SELFTEST_MARK = "SYNTHETIC SELFTEST: injected {off} m depth offset, not a Gate 1 result"
NOTE = "depth is anchored to the table plane over the board footprint: board corners are not fully independent"


@dataclass(frozen=True)
class Landmark:
    """``world`` (3,) m known position, seen at pixel ``(u, v)`` of the undistorted view of camera ``cam``."""

    name: str
    cam: int
    u: float
    v: float
    world: np.ndarray


def board_landmarks(bundle: AcqBundle, spec: BoardSpec | None = None, min_corners: int = MIN_BOARD_CORNERS) -> list[Landmark]:
    """ChArUco corners detected in each camera's undistorted frame of ``bundle`` (cameras with fewer than
    ``min_corners`` corners contribute nothing). Raises ``ValueError`` if the bundle has no frames or no board spec."""
    spec = spec or bundle.board
    if spec is None:
        raise ValueError("the bundle has no board spec; pass --landmarks or acquire with a stored calibration")
    if not bundle.rgb:
        raise ValueError("the bundle has no camera frames to detect the board in")
    known = board_corners_world(spec)
    out: list[Landmark] = []
    for c in sorted(bundle.rgb):
        corners, ids = detect_charuco(spec, np.ascontiguousarray(bundle.rgb[c][..., ::-1]))
        if len(ids) < min_corners:
            continue
        out += [Landmark(f"board_corner_{int(i)}", c, float(p[0]), float(p[1]), known[int(i)]) for p, i in zip(corners, ids)]
    return out


def read_landmarks(path: str | Path) -> list[Landmark]:
    """Landmarks from a CSV with columns ``name,x,y,z,cam,u,v``."""
    rows = read_csv(path)
    need = {"name", "x", "y", "z", "cam", "u", "v"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    return [Landmark(r["name"], int(r["cam"]), float(r["u"]), float(r["v"]), np.array([float(r["x"]), float(r["y"]), float(r["z"])]))
            for r in rows]  # fmt: skip


def landmark_errors(bundle: AcqBundle, landmarks: list[Landmark]) -> list[dict[str, Any]]:
    """Reconstruct every landmark from the frozen depth and compare with its known position. Rows ``LANDMARK_FIELDS``;
    ``status`` is ``ok`` or the click reason (``hole``, ``outside``, ``no_depth``) when no point could be reconstructed
    (``err_m`` is then NaN)."""
    snap = bundle.snapshot()
    rows = []
    for lm in landmarks:
        r = resolve_click(snap, lm.cam, lm.u, lm.v)
        row: dict[str, Any] = {"name": lm.name, "cam": lm.cam, "u": lm.u, "v": lm.v, "x_known": lm.world[0],
                               "y_known": lm.world[1], "z_known": lm.world[2], "status": r.reason}  # fmt: skip
        if r.point is None:
            row.update(x_recon=float("nan"), y_recon=float("nan"), z_recon=float("nan"), err_m=float("nan"), err_z_m=float("nan"))
        else:
            d = r.point - lm.world
            row.update(x_recon=r.point[0], y_recon=r.point[1], z_recon=r.point[2], err_m=float(np.linalg.norm(d)), err_z_m=float(d[2]))
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], target_m: float = TARGET_M) -> list[dict[str, Any]]:
    """M6 summary rows (``SUMMARY_FIELDS``) for scope ``all`` and each camera: median / mean / p95 / max 3-D error [cm],
    median signed Z error [cm], reconstructed fraction, and the verdict ``median < target`` (PRD M6: 3 cm)."""
    out: list[dict[str, Any]] = []
    for scope in ["all"] + [f"cam{c}" for c in sorted({r["cam"] for r in rows})]:
        sub = [r for r in rows if scope == "all" or f"cam{r['cam']}" == scope]
        ok = [r for r in sub if r["status"] == "ok"]
        e = np.array([r["err_m"] for r in ok]) * 100.0
        n = len(sub)

        def add(metric: str, value: float, unit: str, note: str = "", scope: str = scope, n: int = n) -> None:
            out.append({"metric": metric, "scope": scope, "value": value, "unit": unit, "n": n, "note": note})

        add("landmarks_total", float(n), "count")
        add("reconstructed_fraction", len(ok) / n if n else float("nan"), "-", "landmarks whose pixel had valid frozen depth")
        add("err_median_cm", percentile(e, 50), "cm", NOTE)
        add("err_mean_cm", float(e.mean()) if e.size else float("nan"), "cm")
        add("err_p95_cm", percentile(e, 95), "cm")
        add("err_max_cm", float(e.max()) if e.size else float("nan"), "cm")
        add("err_z_median_cm", percentile([r["err_z_m"] * 100.0 for r in ok], 50), "cm", "signed, reconstructed minus known")
        add("meets_m6", float(bool(e.size and np.median(e) < target_m * 100.0)), "bool", f"median error < {target_m * 100:.0f} cm")
    return out


def selftest_bundle(offset_m: float, seed: int = 3) -> AcqBundle:
    """# MOCK: synthetic bundle for the self-test and tests: the default ChArUco board lying at the world origin seen by
    two close cameras (0.55 m, 50 deg elevation, azimuth -30 and +40 deg; synthetic-rig intrinsics), frozen depth =
    exact table-plane Z-depth + ``offset_m`` (a known depth error along the camera Z axis)."""
    from livefire.calib.synthetic_board import SyntheticBoardCamera
    from livefire.common.geometry import AABB, CameraCalibration, look_at
    from livefire.scene.synthetic import load_cameras

    spec, rig = BoardSpec(), load_cameras()[0]
    cams = {}
    for i, az in enumerate((-30.0, 40.0)):
        a, e = np.radians(az), np.radians(50.0)
        pos = 0.55 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        cams[i] = CameraCalibration(i, rig.width, rig.height, rig.K, rig.dist, look_at(pos, [0.0, 0.0, 0.0]), rig.calib_version)
    rng = np.random.default_rng(seed)
    depth, hole, rgb = {}, {}, {}
    for c, cal in cams.items():
        gray = SyntheticBoardCamera(spec, cal.K, cal.dist, (cal.width, cal.height)).render(cal.T_world_from_camera, rng)
        rgb[c] = np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))
        # exact plane z = 0 Z-depth per pixel: t = -c_z / (R d)_z for rays d = K^-1 [u v 1]
        u, v = np.meshgrid(np.arange(cal.width, dtype=np.float64), np.arange(cal.height, dtype=np.float64))
        d = (np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(cal.K).T) @ cal.T_world_from_camera[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -cal.center_world[2] / d[..., 2]
        ok = np.isfinite(t) & (t > 0)
        depth[c] = np.where(ok, t + offset_m, 0.0).astype(np.float32)
        hole[c] = ~ok
    return AcqBundle(cams, depth, hole, rgb, AABB(np.array([-0.8, -0.6, -0.05]), np.array([0.8, 0.6, 0.6])), 0.01, spec,
                     f"selftest board, plane depth + {offset_m} m (synthetic)")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--landmarks", default=None, help="CSV name,x,y,z,cam,u,v instead of the board corners")
    ap.add_argument("--selftest", action="store_true", help="run on a synthetic board with a known depth offset")
    ap.add_argument("--offset-m", type=float, default=0.02, help="--selftest: injected depth offset [m]")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", default="eval/results")
    a = ap.parse_args(argv)
    try:
        bundle = selftest_bundle(a.offset_m) if a.selftest else resolve_bundle(a)
        lms = read_landmarks(a.landmarks) if a.landmarks else board_landmarks(bundle)
        if not lms:
            print("no landmarks: the board was not found (>= 8 corners) in any camera frame. Make sure the board is flat on "
                  "the table, fully visible, unmoved since `calib extrinsics`, then re-acquire.", file=sys.stderr)
            return 1
        rows = landmark_errors(bundle, lms)
    except (ValueError, OSError) as e:
        print(f"alignment failed: {e}", file=sys.stderr)
        return 2
    label = a.label or ("selftest" if a.selftest else "run")
    summ = summarize(rows)
    median_cm = next(r["value"] for r in summ if r["scope"] == "all" and r["metric"] == "err_median_cm")
    met = next(r["value"] for r in summ if r["scope"] == "all" and r["metric"] == "meets_m6")
    for r in summ:
        r["label"] = bundle.label
        if a.selftest:
            r["note"] = SELFTEST_MARK.format(off=a.offset_m) + (f" | {r['note']}" if r["note"] else "")
            if r["metric"] == "meets_m6":
                r["value"] = ""  # a self-test never carries a verdict
    out = Path(a.out_dir)
    write_csv(out / f"alignment_{label}_landmarks.csv", LANDMARK_FIELDS, rows)
    path = write_csv(out / f"alignment_{label}.csv", SUMMARY_FIELDS, summ)
    for s in summ:
        if s["metric"] in ("err_median_cm", "err_p95_cm", "reconstructed_fraction"):
            print(f"{s['scope']:5s} {s['metric']:22s} {s['value']:.3f} {s['unit']}  (n={s['n']})")
    p95 = next(r["value"] for r in summ if r["scope"] == "all" and r["metric"] == "err_p95_cm")
    tag = " [SYNTHETIC SELFTEST, not a Gate 1 result]" if a.selftest else ""
    print(f"M6 ({bundle.label}){tag}: median error {median_cm:.2f} cm, p95 {p95:.2f} cm "
          f"(target median < {TARGET_M * 100:.0f} cm): {'MET' if met else 'NOT MET'}")  # fmt: skip
    print("  caveat: board corners lie on the table plane the depth is anchored to, so this is not a fully independent "
          "check; add elevated landmarks with --landmarks")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
