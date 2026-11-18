"""Spike: board-free metric geometry from ONE real frame with Depth Pro (metric depth + focal length)."""
import sys
import time

import cv2
import numpy as np
import torch
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

src, out = sys.argv[1], sys.argv[2]
c = cv2.VideoCapture(src); ok, bgr = c.read(); c.release()
H, W = bgr.shape[:2]
proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", cache_dir="data/models")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", cache_dir="data/models", torch_dtype=torch.float16).to("cuda").eval()
inp = proc(images=bgr[..., ::-1].copy(), return_tensors="pt").to("cuda")
inp = {k: v.half() if v.is_floating_point() else v for k, v in inp.items()}
for i in range(3):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad(): o = model(**inp)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) * 1e3
pp = proc.post_process_depth_estimation(o, target_sizes=[(H, W)])[0]
Z = pp["predicted_depth"].float().cpu().numpy(); f = float(pp["focal_length"]); fov = 2 * np.degrees(np.arctan(W / 2 / f))
print(f"frame {W}x{H}  latency {dt:.0f} ms  VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MB  focal {f:.0f} px (hfov {fov:.1f} deg)  depth p5/p50/p95 {np.percentile(Z,5):.2f}/{np.median(Z):.2f}/{np.percentile(Z,95):.2f} m")
# back-project, RANSAC the dominant horizontal-ish plane in the lower half (table/floor)
v, u = np.mgrid[0:H, 0:W]
P = np.stack([(u - W / 2) * Z / f, (v - H / 2) * Z / f, Z], -1).reshape(-1, 3)
sel = (v.ravel() > H * 0.45); Q = P[sel][::7]; rng = np.random.default_rng(0); best = (0, None)
for _ in range(400):
    a = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(a[1] - a[0], a[2] - a[0]); nn = np.linalg.norm(n)
    if nn < 1e-9: continue
    n /= nn; d = -n @ a[0]; inl = np.abs(Q @ n + d) < 0.01
    if inl.sum() > best[0]: best = (inl.sum(), (n, d))
n, d = best[1]; dist = np.abs(P @ n + d).reshape(H, W); mask = dist < 0.01
print(f"largest plane in lower half: {mask.mean()*100:.1f}% of pixels within 1 cm; camera height above it {abs(d):.2f} m")
vis = bgr.copy(); vis[mask] = (0.5 * vis[mask] + [0, 120, 0]).astype(np.uint8)
dz = cv2.applyColorMap(cv2.convertScaleAbs(Z, alpha=255 / np.percentile(Z, 98)), cv2.COLORMAP_TURBO)
cv2.imwrite(out, np.concatenate([vis, dz], 1))
