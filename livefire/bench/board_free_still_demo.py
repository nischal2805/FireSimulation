"""Prototype (roadmap M1+M2, docs/REPLAN_LIVE_FIERY.md): board-free fire on ONE still frame of a real video.

Depth Pro gives metric depth and the focal length from the frame (no ChArUco board); the dominant plane (RANSAC on the
lower half) defines the world frame (+Z = plane normal toward the camera, origin = plane inlier centroid). The frame is
played as a still clip through the normal acquisition pipeline (fusion, surfels, objects, snapshot), then the
simulation is ignited at a clicked pixel and composited over the frame. Writes ``<out>/fire.mp4`` and a few PNGs.

Usage (from ``livefire/``)::

    uv run python bench/board_free_still_demo.py VIDEO --click U V [--frame N] [--seconds S] [--radius R] --out DIR
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import DepthProForDepthEstimation, DepthProImageProcessor

from livefire.app.build import build_sim
from livefire.capture.file_source import VideoFileSource
from livefire.capture.types import Frame
from livefire.common.config import load_params, load_yaml
from livefire.common.geometry import CameraCalibration
from livefire.perception.depth.base import depth_result_from_z
from livefire.render.compositor import Compositor
from livefire.scene.acquire import acquire_with_report
from livefire.sim.materials import MaterialTable

VERSION = 1


def depth_pro(rgb: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    """Metric Z-depth (H,W) m and focal length px for an RGB uint8 frame; the model is freed afterwards."""
    proc = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf", cache_dir="data/models")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", cache_dir="data/models",
                                                       torch_dtype=torch.float16).to(device).eval()  # fmt: skip
    inp = {k: (v.half() if v.is_floating_point() else v).to(device) for k, v in proc(images=rgb, return_tensors="pt").items()}
    with torch.no_grad():
        out = model(**inp)
    pp = proc.post_process_depth_estimation(out, target_sizes=[rgb.shape[:2]])[0]
    z, f = pp["predicted_depth"].float().cpu().numpy(), float(pp["focal_length"])
    del model, out
    torch.cuda.empty_cache()
    return z, f


def plane_frame(P: np.ndarray, rows: np.ndarray, H: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC the dominant plane of the lower image half; return (T_world_from_camera 4x4, inlier mask over P)."""
    rng = np.random.default_rng(seed)
    Q = P[rows > H * 0.45][::7]
    best, model = -1, None
    for _ in range(500):
        a = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a[0]
        k = int((np.abs(Q @ n + d) < 0.01).sum())
        if k > best:
            best, model = k, (n, d)
    n, d = model
    inl = np.abs(P @ n + d) < 0.01
    if d < 0:  # +Z toward the camera (camera at the origin has signed distance d)
        n, d = -n, -d
    o = P[inl].mean(axis=0)
    o = o - (o @ n + d) * n  # onto the plane
    ex = np.array([1.0, 0.0, 0.0]) - n[0] * n  # camera x projected onto the plane
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    R_wc = np.stack([ex, ey, n])  # rows: world axes in camera coordinates
    T = np.eye(4)  # p_w = R_wc (p_c - o)  =>  T_world_from_camera = [R_wc | -R_wc o]
    T[:3, :3] = R_wc
    T[:3, 3] = -R_wc @ o
    return T, inl


class FixedDepth:
    """DepthBackend returning a precomputed metric depth for every frame (the scene is still)."""

    device = "cpu"
    peak_vram_allocated_mb = 0.0
    peak_vram_reserved_mb = 0.0

    def __init__(self, z: np.ndarray) -> None:
        self.z = z

    def warmup(self, shape: tuple[int, int]) -> None:
        pass

    def infer(self, frame: Frame):
        return depth_result_from_z(self.z, np.ones_like(self.z, bool), 0.9, frame.calib_version)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--click", type=int, nargs=2, required=True, metavar=("U", "V"))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--radius", type=float, default=0.08)
    ap.add_argument("--material", default=None, help="material for the clicked object (stand-in for M5 materials)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = str(load_yaml("app").get("device", "cpu"))

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {a.frame} of {a.video}")
    H, W = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[..., ::-1])

    t0 = time.perf_counter()
    z, f = depth_pro(rgb, dev if dev.startswith("cuda") else "cpu")
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - W / 2) * z / f, (v - H / 2) * z / f, z], -1).reshape(-1, 3)
    T, inl = plane_frame(P, v.ravel(), H)
    print(f"depth pro: focal {f:.0f} px, depth median {np.median(z):.2f} m, plane covers {inl.mean() * 100:.1f}% "
          f"({time.perf_counter() - t0:.1f} s)")  # fmt: skip
    K = np.array([[f, 0.0, W / 2 - 0.5], [0.0, f, H / 2 - 0.5], [0.0, 0.0, 1.0]])
    cal = CameraCalibration(0, W, H, K, np.zeros(5), T, VERSION)

    still = out / "still.mp4"  # the frozen frame as a clip, so the normal acquisition path is used unchanged
    wr = cv2.VideoWriter(str(still), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for _ in range(30 * 6):
        wr.write(bgr)
    wr.release()
    src = VideoFileSource(0, still, realtime=True, loop=True)
    src.start()
    try:
        res = acquire_with_report({0: src}, {0: cal}, {0: FixedDepth(z)}, version=VERSION, device=dev,
                                  progress=lambda s: print("  acquire:", s))  # fmt: skip
    finally:
        src.stop()
    snap = res.snapshot
    names = {o.obj_id: (o.name, o.material) for o in snap.objects} if hasattr(snap, "objects") else {}
    print(f"snapshot: {len(snap.pos)} surface elements, objects {names}")

    zc = float(z[a.click[1], a.click[0]])
    pc = np.array([(a.click[0] - K[0, 2]) * zc / f, (a.click[1] - K[1, 2]) * zc / f, zc, 1.0])
    pw = (T @ pc)[:3]
    mats = MaterialTable.load()
    sim = build_sim(snap, mats, load_params("sim"), load_yaml("app"))
    oid = int(snap.obj_id[np.argmin(np.linalg.norm(snap.pos - pw, axis=1))])  # object under the click
    if a.material:
        sim.set_material(oid, mats.material_id(a.material))
    n_lit = sim.ignite(pw, a.radius)
    print(f"click ({a.click[0]},{a.click[1]}) -> world {np.round(pw, 3)} m, object {oid}"
          f"{' as ' + a.material if a.material else ''}, {n_lit} elements lit")  # fmt: skip

    comp = Compositor(snap, device=dev)
    frame = Frame(0, rgb, time.perf_counter_ns(), 0, VERSION)
    vw = cv2.VideoWriter(str(out / "fire.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    n_frames, steps = int(a.seconds * 30), max(1, round((1 / 30) / sim.dt_sim))
    t1 = time.perf_counter()
    for i in range(n_frames):
        sim.step(steps)
        img = comp.render(0, frame, sim)[..., ::-1]
        vw.write(img)
        if i in (n_frames // 4, n_frames // 2, n_frames - 1):
            cv2.imwrite(str(out / f"fire_{i:04d}.png"), img)
    vw.release()
    wall = time.perf_counter() - t1
    print(f"rendered {n_frames} frames ({a.seconds} s simulated) in {wall:.1f} s -> {out / 'fire.mp4'}")


if __name__ == "__main__":
    main()
