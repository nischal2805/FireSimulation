"""``python -m livefire.app calib {board,intrinsics,extrinsics}``: ChArUco calibration workflow (README: Calibration).

The interactive loops (OpenCV windows, USB capture) are thin layers over tested pure logic in ``livefire.calib``;
``--from-dir`` runs the same fit/store path headless from saved images. Pathlib only (Windows 11 friendly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from livefire.calib.board import BoardSpec, detect_charuco, load_board_spec, make_board, render_board_png
from livefire.calib.extrinsics import ExtrinsicsResult, estimate_extrinsics
from livefire.calib.intrinsics import (
    IMAGE_EXT,
    CaptureGate,
    CoverageMap,
    IntrinsicsResult,
    View,
    fit_intrinsics,
    load_views_from_dir,
)
from livefire.calib.store import DEFAULT_ROOT, load_intrinsics, save_calibration, save_intrinsics
from livefire.common.config import load_yaml


def run_intrinsics(spec: BoardSpec, views: list[View], size: tuple[int, int], cam_id: int, root: Path,
                   images: dict[str, np.ndarray], cfg: dict) -> IntrinsicsResult:
    """Fit + save intrinsics for ``cam_id`` and print the held-out report. Raises ValueError with guidance."""
    ic = cfg["intrinsics"]
    res = fit_intrinsics(spec, views, size, cam_id=cam_id, min_views=int(ic["min_views"]),
                         holdout_fraction=float(ic["holdout_fraction"]), seed=int(ic["seed"]),
                         fix_k3=bool(ic["fix_k3"]), alpha=float(cfg["undistort"]["alpha"]))
    path = save_intrinsics(root, res, spec, images)
    print(res.summary())
    target = float(ic["target_rms_px"])
    verdict = "OK" if res.rms_heldout < target else f"ABOVE the {target} px target: recapture with more tilt/coverage"
    print(f"held-out RMS {res.rms_heldout:.3f} px (PRD 7.1 target < {target}): {verdict}\nsaved {path}")
    return res


def run_extrinsics(spec: BoardSpec, frames: dict[int, np.ndarray], root: Path, cfg: dict) -> tuple[int, ExtrinsicsResult]:
    """Detect the board in one frame per camera, estimate world poses, store a new calib version."""
    ec = cfg["extrinsics"]
    intr, views = {}, {}
    for cid, img in frames.items():
        intr[cid], stored_spec = load_intrinsics(root, cid)
        if stored_spec != spec:
            raise ValueError(f"cam{cid}: intrinsics were calibrated with a different board {stored_spec}; "
                             "update configs/calibration.yaml or recalibrate")
        if (img.shape[1], img.shape[0]) != (intr[cid].width, intr[cid].height):
            raise ValueError(f"cam{cid}: frame is {img.shape[1]}x{img.shape[0]} but intrinsics are for "
                             f"{intr[cid].width}x{intr[cid].height}")
        c, ids = detect_charuco(spec, img)
        views[cid] = View(c, ids)
    ext = estimate_extrinsics(spec, intr, views, min_corners=int(ec["min_corners"]),
                              max_reproj_rms_px=float(ec["max_reproj_rms_px"]),
                              max_consistency_m=float(ec["max_consistency_m"]))
    ver = save_calibration(root, spec, intr, ext, frames)
    print(ext.summary())
    print(f"saved calib_v{ver:03d} in {root} (active)")
    return ver, ext


def _open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    src: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (isinstance(src, int) and sys.platform == "win32") else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _capture_intrinsics_live(spec: BoardSpec, source: str, size: tuple[int, int], cfg: dict):  # pragma: no cover
    """Window loop: detection overlay, coverage heat-map, auto-capture. Keys: s = capture now, f/q = finish."""
    ic, cc = cfg["intrinsics"], cfg["intrinsics"]["capture"]
    cap = _open_camera(source, *size)
    det = cv2.aruco.CharucoDetector(make_board(spec))
    views: list[View] = []
    images: dict[str, np.ndarray] = {}
    cov = CoverageMap(size, tuple(cc["grid"]))
    gate = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if gate is None:
                size = (w, h)
                cov = CoverageMap(size, tuple(cc["grid"]))
                gate = CaptureGate(size, int(ic["min_corners"]), float(cc["still_px"]), int(cc["still_frames"]),
                                   float(cc["min_pose_diff"]))
            c, ids = detect_charuco(spec, frame, detector=det)
            view = View(c, ids, f"view_{len(views):03d}.png") if len(ids) else None
            got = gate.update(view)
            if got is not None:
                views.append(got)
                images[got.name] = frame.copy()
                cov.add(got.corners)
            vis = cv2.addWeighted(frame, 0.7, cov.heatmap(), 0.3, 0)
            if len(ids):
                cv2.aruco.drawDetectedCornersCharuco(vis, c.reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1))
            cv2.putText(vis, f"views {len(views)}/{ic['min_views']}  coverage {cov.covered_fraction:.0%}  "
                        "[s]=capture [f]=finish", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("calib intrinsics", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("f"), ord("q")):
                break
            if k == ord("s") and view is not None:
                views.append(view)
                images[view.name] = frame.copy()
                cov.add(view.corners)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return views, size, images


def _capture_extrinsics_live(spec: BoardSpec, sources: dict[int, str], size: tuple[int, int]):  # pragma: no cover
    """One frame per camera: SPACE grabs when the board is detected, q aborts."""
    frames: dict[int, np.ndarray] = {}
    det = cv2.aruco.CharucoDetector(make_board(spec))
    for cid, src in sources.items():
        cap = _open_camera(src, *size)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                c, ids = detect_charuco(spec, frame, detector=det)
                vis = frame.copy()
                if len(ids):
                    cv2.aruco.drawDetectedCornersCharuco(vis, c.reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1))
                cv2.putText(vis, f"cam{cid}: {len(ids)} corners  [SPACE]=take [q]=abort", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("calib extrinsics", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    raise KeyboardInterrupt("aborted")
                if k == ord(" ") and len(ids):
                    frames[cid] = frame.copy()
                    break
        finally:
            cap.release()
    cv2.destroyAllWindows()
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="livefire.app calib", description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="calibration store (default data/calib)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="write the print-ready board PNG")
    b.add_argument("--out", default="board.png")
    b.add_argument("--dpi", type=int, default=300)
    i = sub.add_parser("intrinsics", help="calibrate one camera (held-out validated)")
    i.add_argument("--cam", type=int, required=True)
    i.add_argument("--from-dir", default=None, help="use saved images instead of the live camera")
    i.add_argument("--source", default=None, help="USB index or URL (default: --cam as USB index)")
    i.add_argument("--width", type=int, default=1280)
    i.add_argument("--height", type=int, default=720)
    e = sub.add_parser("extrinsics", help="world poses from the board lying flat on the table")
    e.add_argument("--cams", default="0,1", help="comma-separated camera ids")
    e.add_argument("--from-dir", default=None, help="folder with cam<N>.png per camera instead of live capture")
    e.add_argument("--source", action="append", default=[], metavar="N=SRC", help="override camera source")
    e.add_argument("--width", type=int, default=1280)
    e.add_argument("--height", type=int, default=720)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(a.root)
    cfg = load_yaml("calibration")
    spec = load_board_spec(cfg)

    try:
        if a.cmd == "board":
            img, info = render_board_png(spec, a.dpi)
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img)
            bw, bh = info["board_mm"]
            print(f"wrote {out} ({img.shape[1]}x{img.shape[0]} px @ {a.dpi} dpi): page {info['page_mm'][0]:.0f}x"
                  f"{info['page_mm'][1]:.0f} mm; board {bw:.1f}x{bh:.1f} mm, squares {info['square_mm']:.1f} mm, "
                  f"markers {info['marker_mm']:.1f} mm.\nPRINT AT 100 % SCALE (no fit-to-page), then MEASURE one square "
                  "with a ruler and set board.square_length_m in configs/calibration.yaml.")
            return 0
        if a.cmd == "intrinsics":
            ic = cfg["intrinsics"]
            if a.from_dir:
                views, size, used = load_views_from_dir(spec, Path(a.from_dir), int(ic["min_corners"]))
                images = {p.name: cv2.imread(str(p)) for p in used}
            else:
                views, size, images = _capture_intrinsics_live(spec, a.source or str(a.cam), (a.width, a.height), cfg)
            run_intrinsics(spec, views, size, a.cam, root, images, cfg)
            return 0
        cams = [int(x) for x in a.cams.split(",")]
        if a.from_dir:
            frames = {}
            for c in cams:
                cand = [p for p in Path(a.from_dir).iterdir() if p.stem == f"cam{c}" and p.suffix.lower() in IMAGE_EXT]
                if not cand:
                    raise ValueError(f"{a.from_dir}: no cam{c}.png/jpg")
                frames[c] = cv2.imread(str(cand[0]))
        else:
            src = {c: str(c) for c in cams} | {int(s.split("=")[0]): s.split("=", 1)[1] for s in a.source}
            frames = _capture_extrinsics_live(spec, src, (a.width, a.height))
        run_extrinsics(spec, frames, root, cfg)
        return 0
    except (ValueError, RuntimeError, KeyboardInterrupt) as ex:
        print(f"calibration failed: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
