"""Segmentation backend spike (P3-001, ADR-0007; PRD 7.3, 11.6): mask quality, category accuracy, latency, VRAM.

Usage: python bench/p3_segment_spike.py [--device cpu|cuda:0] [--backends gdino_sam2,sam3,mask2former]
                                        [--clips DIR] [--coco-dir DIR] [--coco-images N] [--repeats 5] [--out CSV]

Inputs:
(a) ``table_v1`` renders from both synthetic cameras (sanity only: flat shading, out of the models' domain; GT =
    exact per-pixel ``obj_id``, categories via ``configs/segment.yaml: category_map``).
(b) The user's recorded clips when present: ``<DIR>/cam<N>.mp4`` (the active calibration in ``--calib-root`` undistorts
    them) plus hand-labelled keyframes in ``<DIR>/labels/``: for each labelled frame, ``cam<N>_f<FRAME>.png`` (a
    single-channel 8/16-bit PNG of instance ids, 0 = background, ids 1..K) and ``cam<N>_f<FRAME>.yaml``
    (``categories: {1: "wooden block", 2: "cloth", ...}``, phrases from the vocabulary; ``FRAME`` = 0-based index in the
    clip). Frames without both files are ignored. NOT RUN here (no clips exist in the development container).
(c) Optionally COCO val2017 images (``--coco-dir``, built by ``--fetch-coco``): images + ``instances.json`` (COCO format
    subset) + ``manifest.json`` (per-image Flickr licence; annotations CC BY 4.0). Skipped when absent. COCO classes
    are not the project vocabulary, so category accuracy is not computed for (c); mask recall/IoU are.

Metrics per image (Hungarian matching of GT and predicted instances on mask IoU): mean matched IoU over GT objects
(unmatched count as 0), recall at IoU 0.5, category accuracy among the matches with IoU >= 0.5 (synthetic and clips),
number of predicted instances. Latency = ``backend.segment`` only: median / p95 of ``--repeats`` timed calls (default 5)
after one warm-up call; peak VRAM from the backend counters (CUDA only). CPU timings are labelled ``gpu=none``,
``framework=torch-<ver>-cpu``. ``sam3`` is gated (manual approval, Meta SAM License): recorded as ``skipped: gated`` in
the manifest unless it loads. Output: ``bench/results/segment_spike_cpu_<date>.csv`` (Appendix C schema, metrics in
``notes``), ``..._frames.csv`` (per image) and ``..._manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from livefire.capture.types import Frame
from livefire.common.bench_csv import BenchmarkRecorder
from livefire.common.config import load_yaml
from livefire.perception.segment import create_segmentation_backend, load_segment_config
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

ROOT = Path(__file__).resolve().parents[1]
# images.cocodataset.org fails TLS verification (hostname mismatch, checked 2026-11-15), so plain http is used and
# integrity is pinned by the sha256 values recorded in the manifest (never disable certificate checks).
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def instance_masks(ids: np.ndarray, ignore=(-1,)) -> dict[int, np.ndarray]:
    """Boolean mask per instance id of an id image (ids in ``ignore`` are background). Default: only -1 (backend
    output numbers instances from 0); hand-labelled PNGs use 0 as background and pass ``ignore=(0,)``."""
    return {int(i): ids == i for i in np.unique(ids) if int(i) not in ignore}


def match_metrics(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray], gt_cat: dict[int, str] | None,
                  pred_cat: dict[int, str]) -> dict[str, float]:
    """Hungarian matching on mask IoU. Returns mean IoU over GT (unmatched = 0), recall@0.5, category accuracy among
    matches with IoU >= 0.5 (NaN without GT categories or matches) and the number of predictions."""
    g, p = list(gt), list(pred)
    iou = np.zeros((len(g), len(p)))
    for a, gi in enumerate(g):
        for b, pi in enumerate(p):
            u = (gt[gi] | pred[pi]).sum()
            iou[a, b] = (gt[gi] & pred[pi]).sum() / u if u else 0.0
    per_gt = np.zeros(len(g))
    hits, correct = 0, 0
    if g and p:
        rows, cols = linear_sum_assignment(-iou)
        for a, b in zip(rows, cols, strict=True):
            per_gt[a] = iou[a, b]
            if iou[a, b] >= 0.5:
                hits += 1
                correct += int(gt_cat is not None and gt_cat[g[a]] == pred_cat.get(p[b]))
    return {"miou": float(per_gt.mean()) if g else float("nan"), "recall50": hits / len(g) if g else float("nan"),
            "cat_acc": correct / hits if (gt_cat is not None and hits) else float("nan"), "n_gt": len(g),
            "n_pred": len(p), "matched50": hits}


# --- inputs ----------------------------------------------------------------------------------------------------
def synthetic_items():
    """(name, rgb, gt_masks, gt_cats, calib_version) for both table_v1 cameras."""
    scene = build_table_v1()
    cmap = load_segment_config()["category_map"]
    info = scene.objects_table()
    out = []
    for cid, cam in load_cameras().items():
        v = render_view(scene, cam)
        out.append((f"table_v1_cam{cid}", v.rgb, instance_masks(v.obj_id, ignore=(-1,)),
                    {o: cmap[info[o][0]] for o in info}, cam.calib_version))
    return out


def clip_items(clips: Path, calib_root: Path):
    """Hand-labelled keyframes of the user's clips (format in the module docstring)."""
    from livefire.calib.store import load_calibration

    labels = clips / "labels"
    if not labels.is_dir():
        return []
    cs = load_calibration(calib_root)
    items = []
    for png in sorted(labels.glob("cam*_f*.png")):
        stem = png.stem  # cam<N>_f<FRAME>
        cam_id, frame_idx = int(stem[3:].split("_f")[0]), int(stem.split("_f")[1])
        ymlp = png.with_suffix(".yaml")
        vid = clips / f"cam{cam_id}.mp4"
        if not ymlp.is_file() or not vid.is_file() or cam_id not in cs.cameras:
            continue
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            continue
        rec = cs.cameras[cam_id]
        rgb = np.ascontiguousarray(rec.undistorter().apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        ids = cv2.imread(str(png), cv2.IMREAD_UNCHANGED).astype(np.int32)
        cats = {int(k): str(v) for k, v in yaml.safe_load(ymlp.read_text())["categories"].items()}
        items.append((stem, rgb, instance_masks(ids, ignore=(0,)), cats, rec.calibration.calib_version))
    return items


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_coco(out_dir: Path, n_images: int) -> None:
    """Download COCO val2017 annotations (CC BY 4.0, ~250 MB zip, cached) and ``n_images`` images that contain at least
    two non-crowd instances of 4+ categories with area >= 2 % of the image; write ``instances.json`` (subset) and
    ``manifest.json`` (per-image Flickr licence id and URL)."""
    import io
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    zp = out_dir / "ann.zip"
    if not zp.is_file() or zp.stat().st_size < 200_000_000:
        _log(f"downloading {COCO_ANN_URL}")
        urllib.request.urlretrieve(COCO_ANN_URL, zp)
    with zipfile.ZipFile(zp) as z:
        ann = json.load(io.TextIOWrapper(z.open("annotations/instances_val2017.json"), encoding="utf-8"))
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        if not a["iscrowd"]:
            by_img.setdefault(a["image_id"], []).append(a)
    imgs = {i["id"]: i for i in ann["images"]}
    lic = {entry["id"]: entry for entry in ann["licenses"]}
    chosen = []
    for iid in sorted(by_img):
        im = imgs[iid]
        big = [a for a in by_img[iid] if a["area"] >= 0.02 * im["width"] * im["height"]]
        if len(big) >= 2 and len({a["category_id"] for a in big}) >= 2 and len(big) <= 8:
            chosen.append(iid)
        if len(chosen) >= n_images:
            break
    (out_dir / "images").mkdir(exist_ok=True)
    manifest = {"annotations": "COCO 2017 val instances, CC BY 4.0, https://cocodataset.org/#termsofuse",
                "annotations_zip": {"url": COCO_ANN_URL, "sha256": _sha256(zp), "bytes": zp.stat().st_size},
                "images": []}
    sub = {"images": [], "annotations": [], "categories": ann["categories"]}
    for iid in chosen:
        im = imgs[iid]
        p = out_dir / "images" / im["file_name"]
        if not p.is_file():
            urllib.request.urlretrieve(COCO_IMG_URL.format(file_name=im["file_name"]), p)
        sub["images"].append(im)
        sub["annotations"] += [a for a in by_img[iid] if a["area"] >= 0.02 * im["width"] * im["height"]]
        manifest["images"].append({"file": im["file_name"], "flickr_url": im.get("flickr_url"),
                                   "license": lic[im["license"]]["name"], "license_url": lic[im["license"]]["url"], "sha256": _sha256(p),
                                   "url": COCO_IMG_URL.format(file_name=im["file_name"])})
    (out_dir / "instances.json").write_text(json.dumps(sub))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    _log(f"COCO subset: {len(chosen)} images in {out_dir}")


def coco_items(coco_dir: Path, n_images: int):
    """(name, rgb, gt_masks, None, 0) for the COCO subset; polygon/RLE masks rasterised with OpenCV (polygons only)."""
    jp = coco_dir / "instances.json"
    if not jp.is_file():
        return []
    ann = json.loads(jp.read_text())
    items = []
    for im in ann["images"][:n_images]:
        bgr = cv2.imread(str(coco_dir / "images" / im["file_name"]))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        masks = {}
        for a in ann["annotations"]:
            if a["image_id"] != im["id"] or not isinstance(a["segmentation"], list):
                continue
            m = np.zeros((h, w), np.uint8)
            for poly in a["segmentation"]:
                cv2.fillPoly(m, [np.round(np.array(poly).reshape(-1, 2)).astype(np.int32)], 1)
            masks[int(a["id"])] = m.astype(bool)
        items.append((f"coco_{im['id']}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), masks, None, 0))
    return items


# --- evaluation ------------------------------------------------------------------------------------------------
def evaluate(backend, items, label: str, repeats: int):
    rows = []
    first = True
    for name, rgb, gt, gt_cat, cv in items:
        fr = Frame(0, rgb, 0, 0, cv)
        if first:
            backend.warmup(rgb.shape[:2])
            first = False
        backend.segment(fr)  # warm-up on the real input (shapes, caches)
        backend.reset_peak_vram()
        lats = []
        for _ in range(repeats):
            res = backend.segment(fr)
            lats.append(backend.latency_ms)
        pred = instance_masks(res.instance_ids, ignore=(-1,))  # ids start at 0: never drop id 0
        m = match_metrics(gt, pred, gt_cat, res.categories)
        row = {"dataset": label, "frame": name, "latency_median_ms": float(np.median(lats)),
               "latency_p95_ms": float(np.percentile(lats, 95)), "latencies_ms": ";".join(f"{x:.1f}" for x in lats), "peak_vram_alloc_mb": backend.peak_vram_allocated_mb,
               "peak_vram_reserved_mb": backend.peak_vram_reserved_mb, **m,
               "pred_categories": ";".join(sorted(res.categories.values()))}
        rows.append(row)
        _log(f"{label} {name}: mIoU {m['miou']:.3f} recall@0.5 {m['recall50']:.2f} cat {m['cat_acc']:.2f} "
             f"pred {m['n_pred']} lat {row['latency_median_ms']:.0f} ms")
    return rows


def _mean(rows, key):
    v = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def csv_row(rec, rows, name, label, resolution, extra, repeats, device):
    lat = np.array([float(x) for r in rows for x in r["latencies_ms"].split(";")])  # all timed calls pooled
    notes = [f"latency percentiles pooled over {len(lat)} timed calls of {len(rows)} images", f"backend={name}", f"dataset={label}", f"images={len(rows)}", f"miou={_mean(rows, 'miou'):.3f}",
             f"recall50={_mean(rows, 'recall50'):.3f}", f"cat_acc={_mean(rows, 'cat_acc'):.3f}",
             f"n_pred_mean={_mean(rows, 'n_pred'):.1f}", f"n_gt_mean={_mean(rows, 'n_gt'):.1f}", extra]
    rec.add_row(scene=label, resolution=resolution, mode="spike", stage="segment_infer", variant=name,
                warmup_frames=1, samples=len(lat), median_ms=round(float(np.median(lat)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3), p99_ms=round(float(np.percentile(lat, 99)), 3), views=1,
                peak_vram_alloc_mb=rows[0]["peak_vram_alloc_mb"] if device != "cpu" else "",
                peak_vram_reserved_mb=rows[0]["peak_vram_reserved_mb"] if device != "cpu" else "",
                notes="; ".join(notes))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--backends", default="gdino_sam2,sam3,mask2former")
    ap.add_argument("--clips", default=str(ROOT / "data" / "clips"))
    ap.add_argument("--calib-root", default=str(ROOT / "data" / "calib"))
    ap.add_argument("--coco-dir", default=str(ROOT / "data" / "datasets" / "coco_val2017_subset"))
    ap.add_argument("--coco-images", type=int, default=6)
    ap.add_argument("--fetch-coco", action="store_true", help="download the COCO val2017 subset first")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--wall-note", default="", help="free text copied into the manifest (e.g. machine load)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = args.device or str(load_yaml("app")["device"])
    cfg = load_segment_config()
    import torch

    fw = f"torch-{torch.__version__.split('+')[0]}" + ("-cpu" if device == "cpu" else "")
    rec = BenchmarkRecorder(device, framework=fw, workload_version="segment-spike-v1")
    cores = f"CPU only, torch threads={torch.get_num_threads()}" if device == "cpu" else "GPU"
    manifest: dict = {"device": device, "candidates": {}, "coco": None, "wall_note": args.wall_note,
                      "started": datetime.datetime.now().isoformat(timespec="seconds")}
    t_start = time.perf_counter()
    if args.fetch_coco:
        fetch_coco(Path(args.coco_dir), args.coco_images)
    syn = synthetic_items()
    coco_dir = Path(args.coco_dir)
    coco = coco_items(coco_dir, args.coco_images)
    if coco and (coco_dir / "manifest.json").is_file():
        manifest["coco"] = json.loads((coco_dir / "manifest.json").read_text())
    else:
        _log("no COCO subset (run with --fetch-coco): skipping (c)")
    clips = Path(args.clips)
    user = []
    if clips.is_dir():
        try:
            user = clip_items(clips, Path(args.calib_root))
        except Exception as e:  # noqa: BLE001
            _log(f"user clips skipped: {e}")
    if not user:
        _log(f"no labelled user clips in {clips}: skipping (b)")
    all_rows: list[dict] = []
    for name in [b for b in args.backends.split(",") if b]:
        t0 = time.perf_counter()
        try:
            be = create_segmentation_backend(name, cfg, device)
        except Exception as e:  # noqa: BLE001
            gated = "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower()
            manifest["candidates"][name] = {"status": "skipped: gated" if gated else f"not measured: {e}"[:300]}
            _log(f"{name}: {manifest['candidates'][name]['status']}")
            continue
        manifest["candidates"][name] = {"status": "measured", "load_s": round(time.perf_counter() - t0, 1)}
        extra = f"{cores}; repeats={args.repeats}; thresholds box={cfg['box_threshold']['value']}"
        for label, items, res in (("synthetic_table_v1", syn, "1280x720"), ("coco_val2017_subset", coco, "native"),
                                  ("user_clips", user, "native")):
            if not items:
                continue
            rows = evaluate(be, items, label, args.repeats)
            for r in rows:
                r["backend"] = name
            all_rows += rows
            tag = "sanity only (flat-shaded render); " if label.startswith("synthetic") else ""
            csv_row(rec, rows, name, label, res, tag + extra, args.repeats, device)
        be.release()
        del be
    stamp = datetime.date.today().isoformat()
    out = Path(args.out) if args.out else ROOT / "bench" / "results" / f"segment_spike_{'cpu' if device == 'cpu' else 'gpu'}_{stamp}.csv"
    rec.flush(str(out))
    keys = sorted({k for r in all_rows for k in r})
    with out.with_name(out.stem + "_frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    manifest["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    out.with_name(out.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {out}")


if __name__ == "__main__":
    main()
