"""Depth backend spike (P1-004, ADR-0005; PRD 7.2, 11.6): accuracy, coverage, latency, VRAM -> Appendix C CSV.

Usage: python bench/depth_spike.py [--device cpu|cuda:0] [--backends a,b] [--tum-dir DIR] [--tum-frames N]
                                   [--clips DIR] [--out CSV]

Datasets: (a) synthetic table_v1 render (sanity only, exact GT); (b) TUM RGB-D fr3/long_office_household
(CC BY 4.0, https://cvg.cit.tum.de/data/datasets/rgbd-dataset), anchored on the dominant plane fitted to the GT depth
of each frame (the plane stands in for the calibrated table plane; its inlier hull is the table region);
(c) user clips ``<DIR>/cam<N>.mp4`` with the active calibration in ``data/calib`` (no GT: coverage, anchor residual,
latency only). CPU timings are labelled ``gpu=none, framework=torch-<ver>-cpu``. Latency = ``infer_raw`` only (network);
the anchor fit is timed separately in ``notes``. Metrics: MAE (m), AbsRel, valid coverage (anchored valid / GT valid).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from livefire.capture.types import Frame
from livefire.common.bench_csv import BenchmarkRecorder
from livefire.common.config import load_yaml
from livefire.common.errors import DepthError
from livefire.common.geometry import CameraCalibration, make_K
from livefire.perception.depth import (
    AnchorConfig,
    OracleDepthBackend,
    TableRegion,
    apply_anchor,
    fit_anchor,
    table_plane_depth,
)
from livefire.perception.depth.base import load_depth_config
from livefire.perception.depth.hf_backends import create_depth_backend
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

ROOT = Path(__file__).resolve().parents[1]
TUM_SEQ = "rgbd_dataset_freiburg3_long_office_household"
TUM_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
TUM_K = (535.4, 539.2, 320.1, 247.6)  # TUM calibration page, Freiburg 3 RGB (images already undistorted)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def metrics(z: np.ndarray, valid: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray) -> dict[str, float]:
    """MAE (m), AbsRel and valid coverage of ``z`` against ``gt`` over GT-valid pixels."""
    both = valid & gt_valid
    if not both.any():
        return {"mae_m": float("nan"), "absrel": float("nan"), "coverage": 0.0}
    e = np.abs(z[both].astype(np.float64) - gt[both])
    return {"mae_m": float(e.mean()), "absrel": float((e / gt[both]).mean()),
            "coverage": float(both.sum() / max(gt_valid.sum(), 1))}


# --- TUM -------------------------------------------------------------------------------------------------------
def _associate(d: Path) -> list[tuple[str, str]]:
    def read(n: str) -> list[tuple[float, str]]:
        return [(float(t), f) for t, f in (ln.split()[:2] for ln in (d / n).read_text().splitlines()
                                           if ln and not ln.startswith("#"))]

    depth = read("depth.txt")
    dts = np.array([t for t, _ in depth])
    out = []
    for t, f in read("rgb.txt"):
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) < 0.02:
            out.append((f, depth[j][1]))
    return out


def fit_plane_ransac(pts: np.ndarray, rng: np.random.Generator, thr: float = 0.02, iters: int = 300):
    """Dominant plane ``n.p + d = 0`` (unit n) of camera-frame points (N,3); returns (n, d, inlier mask)."""
    best, best_n = None, -1
    for _ in range(iters):
        s = pts[rng.integers(0, len(pts), 3)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ s[0]
        cnt = int((np.abs(pts @ n + d) < thr).sum())
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    for _ in range(3):
        m = np.abs(pts @ n + d) < thr
        c = pts[m].mean(0)
        _, _, vt = np.linalg.svd(pts[m] - c, full_matrices=False)
        n = vt[2]
        d = -n @ c
    return n, d, np.abs(pts @ n + d) < thr


def plane_calibration(gt: np.ndarray, gt_valid: np.ndarray, rng: np.random.Generator):
    """Camera calibration whose world frame has the dominant GT plane at z = 0 (+Z toward the camera), plus the
    table region (convex hull of the plane inliers in world XY) and the plane inlier fraction."""
    h, w = gt.shape
    K = make_K(*TUM_K)
    v, u = np.nonzero(gt_valid)
    z = gt[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    sub = rng.choice(len(pts), min(len(pts), 20000), replace=False)
    n, d, _ = fit_plane_ransac(pts[sub], rng)
    if d < 0:
        n, d = -n, -d  # camera origin at positive height
    x = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])  # camera -> world rotation (rows = world axes in camera coords)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.0, 0.0, d]
    calib = CameraCalibration(0, w, h, K, np.zeros(5), T, 1)
    inl = np.abs(pts @ n + d) < 0.02
    pw = (pts[inl] - (-d * n)) @ R.T
    hull = ConvexHull(pw[:, :2][:: max(1, inl.sum() // 5000)])
    poly = pw[:, :2][:: max(1, inl.sum() // 5000)][hull.vertices]
    return calib, TableRegion(poly), float(inl.mean())


def load_tum(tum_dir: Path, n_frames: int, rng: np.random.Generator):
    """Yield (name, rgb uint8, gt f32 m, gt_valid, calib, region, plane_frac) for frames with a dominant plane
    covering >= 15 % of the GT-valid pixels; evenly spaced candidates over the sequence."""
    d = tum_dir / TUM_SEQ
    pairs = _associate(d)
    cands = np.linspace(100, len(pairs) - 100, n_frames * 4).astype(int)
    got = 0
    for i in cands:
        rgb_f, dep_f = pairs[i]
        bgr = cv2.imread(str(d / rgb_f))
        raw = cv2.imread(str(d / dep_f), cv2.IMREAD_UNCHANGED)
        gt = (raw.astype(np.float32) / 5000.0)
        gt_valid = raw > 0
        calib, region, frac = plane_calibration(gt, gt_valid, rng)
        if frac < 0.15:
            _log(f"TUM frame {i} skipped: dominant plane covers {frac:.1%} (< 15 %)")
            continue
        yield (f"fr3_lo_{i}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.where(gt_valid, gt, 0.0), gt_valid,
               calib, region, frac)
        got += 1
        if got >= n_frames:
            return


# --- evaluation ------------------------------------------------------------------------------------------------
def evaluate(backend, name, kind, frames, anchors: dict[str, AnchorConfig], label: str):
    """frames: iterable of (name, rgb, gt, gt_valid, calib, region, plane_frac). Returns per-frame dict list."""
    rows = []
    first = True
    for fname, rgb, gt, gt_valid, calib, region, frac in frames:
        fr = Frame(0, rgb, 0, 0, calib.calib_version)
        if first:
            backend.warmup(rgb.shape[:2])
            first = False
        raw = backend.infer_raw(fr)
        lat = backend.latency_ms
        row = {"dataset": label, "backend": name, "frame": fname, "latency_ms": lat, "plane_frac": frac,
               "peak_vram_alloc_mb": backend.peak_vram_allocated_mb,
               "peak_vram_reserved_mb": backend.peak_vram_reserved_mb}
        if raw.kind == "metric":
            m = metrics(raw.values, raw.valid, gt, gt_valid)
            row.update({f"raw_{k}": v for k, v in m.items()})
        plane_z, mask = table_plane_depth(calib, region)
        base = row
        for tag, cfg_a in anchors.items():
            row = {**base, "anchor_cfg": tag}
            t0 = time.perf_counter_ns()
            try:
                fit = fit_anchor(raw.values, raw.valid, plane_z, mask, raw.kind, cfg_a)
            except DepthError as e:
                row.update({"status": f"rejected: {e}"})
                rows.append(row)
                continue
            z, valid = apply_anchor(raw, fit, cfg_a.z_max_m)
            row["anchor_ms"] = (time.perf_counter_ns() - t0) / 1e6
            row.update(metrics(z, valid, gt, gt_valid))
            tm = mask & gt_valid
            row["table_mae_m"] = metrics(z, valid, gt, tm)["mae_m"]
            row.update({"status": "ok", "inlier_ratio": fit.inlier_ratio, "scale": fit.scale, "shift": fit.shift,
                        "resid_mae_m": fit.resid_mae_m})
            rows.append(row)
            _log(f"{label} {name} {fname} [{tag}]: MAE {row['mae_m']:.3f} m AbsRel {row['absrel']:.3f} "
                 f"cov {row['coverage']:.2f} inl {fit.inlier_ratio:.2f} lat {lat:.0f} ms")
    return rows


def _agg(rows: list[dict], key: str) -> float:
    v = [r[key] for r in rows if r.get("status") == "ok" and key in r and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def to_csv_row(rec: BenchmarkRecorder, rows_all: list[dict], scene: str, resolution: str, kind: str, extra: str) -> None:
    for tag in dict.fromkeys(r.get("anchor_cfg", "") for r in rows_all):
        _csv_row(rec, [r for r in rows_all if r.get("anchor_cfg", "") == tag], scene, resolution, kind, extra, tag)


def _csv_row(rec, rows, scene, resolution, kind, extra, tag) -> None:
    if not rows:
        return
    lat = np.array([r["latency_ms"] for r in rows])
    ok = [r for r in rows if r.get("status") == "ok"]
    notes = [f"anchor={kind}/{tag}", f"frames_ok={len(ok)}/{len(rows)}", f"mae_m={_agg(rows, 'mae_m'):.4f}",
             f"absrel={_agg(rows, 'absrel'):.4f}", f"table_mae_m={_agg(rows, 'table_mae_m'):.4f}",
             f"coverage={_agg(rows, 'coverage'):.3f}", f"inlier_ratio={_agg(rows, 'inlier_ratio'):.3f}"]
    if "raw_mae_m" in rows[0]:
        notes.append(f"unanchored_mae_m={np.nanmean([r['raw_mae_m'] for r in rows]):.4f}")
        notes.append(f"unanchored_absrel={np.nanmean([r['raw_absrel'] for r in rows]):.4f}")
    notes.append(f"anchor_ms_mean={_agg(rows, 'anchor_ms'):.1f}")
    if extra:
        notes.append(extra)
    rec.add_row(
        scene=scene, resolution=resolution, mode="spike", stage="depth_infer", variant=f"{rows[0]['backend']}+anchor_{tag}",
        warmup_frames=1, samples=len(lat), median_ms=round(float(np.median(lat)), 3),
        p95_ms=round(float(np.percentile(lat, 95)), 3), p99_ms=round(float(np.percentile(lat, 99)), 3),
        views=1, peak_vram_alloc_mb=rows[0]["peak_vram_alloc_mb"] if rec.device != "cpu" else "",
        peak_vram_reserved_mb=rows[0]["peak_vram_reserved_mb"] if rec.device != "cpu" else "",
        notes="; ".join(notes),
    )


def synthetic_frames():
    cam = load_cameras()[0]
    v = render_view(build_table_v1(), cam)
    region = TableRegion.from_footprint((0.8, 0.5), 0.0)  # table top (z = 0 plane), objects on it are outliers
    _, m = table_plane_depth(cam, region)
    return cam, [("table_v1_cam0", v.rgb, v.depth, v.valid, cam, region, float(m.mean()))]


def clip_rows(backend, name, clips: Path, cfg_a: AnchorConfig, step: int = 15, max_frames: int = 10):
    from livefire.calib.store import load_calibration

    cs = load_calibration(Path("data") / "calib")
    region = TableRegion.from_config()
    rows = []
    for cam_id, rec in cs.cameras.items():
        p = clips / f"cam{cam_id}.mp4"
        if not p.is_file():
            continue
        und, cal = rec.undistorter(), rec.calibration
        cap = cv2.VideoCapture(str(p))
        i = 0
        plane_z, mask = table_plane_depth(cal, region)
        first = True
        while len(rows) < max_frames * len(cs.cameras):
            ok, bgr = cap.read()
            if not ok:
                break
            i += 1
            if i % step:
                continue
            rgb = np.ascontiguousarray(und.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if first:
                backend.warmup(rgb.shape[:2])
                first = False
            raw = backend.infer_raw(Frame(cam_id, rgb, 0, i, cal.calib_version))
            row = {"dataset": "user_clips", "backend": name, "frame": f"cam{cam_id}_{i}", "latency_ms": backend.latency_ms,
                   "anchor_cfg": "strict", "plane_frac": float(mask.mean()), "peak_vram_alloc_mb": backend.peak_vram_allocated_mb,
                   "peak_vram_reserved_mb": backend.peak_vram_reserved_mb}
            try:
                fit = fit_anchor(raw.values, raw.valid, plane_z, mask, raw.kind, cfg_a)
                _z, valid = apply_anchor(raw, fit, cfg_a.z_max_m)
                row.update({"status": "ok", "inlier_ratio": fit.inlier_ratio, "resid_mae_m": fit.resid_mae_m,
                            "coverage": float(valid.mean())})
            except DepthError as e:
                row["status"] = f"rejected: {e}"
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--backends", default="da2_metric_small,da2_relative_small")
    ap.add_argument("--tum-dir", default=str(ROOT / "data" / "datasets" / "tum_fr3_long_office"))
    ap.add_argument("--tum-frames", type=int, default=10)
    ap.add_argument("--clips", default=str(ROOT / "data" / "clips"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = args.device or str(load_yaml("app")["device"])
    cfg = load_depth_config()
    cfg_a = AnchorConfig.from_yaml(cfg)
    rng = np.random.default_rng(cfg_a.seed)
    # strict = configs/depth.yaml; lenient = diagnostic sensitivity run (looser RANSAC threshold, lower minimum ratio)
    anchors = {"strict": cfg_a, "lenient": replace(cfg_a, inlier_thresh_rel=0.08, min_inlier_ratio=0.3)}
    import torch

    fw = f"torch-{torch.__version__.split('+')[0]}" + ("-cpu" if device == "cpu" else "")
    rec = BenchmarkRecorder(device, framework=fw, workload_version="depth-spike-v1")
    all_rows: list[dict] = []
    cores = f"CPU only, torch threads={torch.get_num_threads()}" if device == "cpu" else "GPU"
    manifest = {"tum": {"sequence": TUM_SEQ, "url": TUM_URL, "license": "CC BY 4.0",
                        "license_url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
                        "intrinsics": TUM_K, "depth_scale": 5000}}

    tum = list(load_tum(Path(args.tum_dir), args.tum_frames, rng)) if (Path(args.tum_dir) / TUM_SEQ).is_dir() else []
    if not tum:
        _log("TUM sequence missing or no usable frames: skipping (b); see docs/decisions/ADR-0005 for the download")
    manifest["tum"]["frames"] = [t[0] for t in tum]
    _, syn = synthetic_frames()

    # (a) Oracle sanity: exact depth, metric and disparity + anchor
    for kind, kw in (("metric", {}), ("disparity", {"disp_a": 3.0, "disp_b": 0.7})):
        be = OracleDepthBackend(syn[0][4], kind=kind, **kw)
        rows = evaluate(be, f"oracle_{kind}", kind, syn, {"strict": cfg_a}, "synthetic")
        all_rows += rows
        to_csv_row(rec, rows, "synthetic_table_v1", "1280x720", kind, f"MOCK oracle (ground truth); {cores}")

    for name in [b for b in args.backends.split(",") if b]:
        be = create_depth_backend(name, cfg, device)
        be.reset_peak_vram()
        kind = cfg["backends"][name]["kind"]
        extra = f"hf={cfg['backends'][name]['hf_id']}; input_size={cfg['input_size']}; {cores}"
        rows = evaluate(be, name, kind, syn, anchors, "synthetic")
        all_rows += rows
        to_csv_row(rec, rows, "synthetic_table_v1", "1280x720", kind, "sanity only (flat-shaded render); " + extra)
        if tum:
            rows = evaluate(be, name, kind, tum, anchors, "tum")
            all_rows += rows
            to_csv_row(rec, rows, "tum_fr3_long_office_household", "640x480", kind,
                       f"anchor plane = dominant GT plane per frame; {extra}")
        clips = Path(args.clips)
        if clips.is_dir() and any(clips.glob("cam*.mp4")):
            try:
                rows = clip_rows(be, name, clips, cfg_a)
                all_rows += rows
                to_csv_row(rec, rows, "user_clips", "native", kind, "no ground truth: coverage/anchor residual only; " + extra)
            except Exception as e:  # noqa: BLE001
                _log(f"user clips skipped: {e}")
        else:
            _log(f"no user clips in {clips}: skipping (c)")
        del be

    stamp = datetime.date.today().isoformat()
    out = Path(args.out) if args.out else ROOT / "bench" / "results" / f"depth_spike_{'cpu' if device == 'cpu' else 'gpu'}_{stamp}.csv"
    rec.flush(str(out))
    frames_csv = out.with_name(out.stem + "_frames.csv")
    keys = sorted({k for r in all_rows for k in r})
    with frames_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    out.with_name(out.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {out}, {frames_csv}")


if __name__ == "__main__":
    main()
