"""Two-view fire compositor: volume ray march + char/removed overlay over the live frame (PRD 7.4).

``Compositor`` implements ``RendererBackend``: ``render(cam_id, frame, fire) -> uint8 (H,W,3)``
sRGB, host, C-contiguous, caller-owned, synchronised on return. It owns its device mirrors of the
snapshot geometry (ray directions, occlusion depth) and never writes the ``FireStateView`` fields.

Pipeline per view: march at ``volume_scale`` resolution (ROI AND active-cell AABB AND occlusion
clipped) -> bilinear upsample of (T_vol, L_vol) -> base = linear(frame) darkened by char, mixed to ash
by removed -> ``out = base * T_vol + tonemap(exposure * L_vol)``, ``tonemap(x) = x / (1 + x)`` -> sRGB.

``python -m livefire.render.compositor --demo`` writes two composited PNGs (prescribed blob).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import warp as wp
from numpy.typing import NDArray

from livefire.capture.types import Frame
from livefire.common.bench_csv import device_label
from livefire.common.config import Params, load_params, load_yaml
from livefire.common.geometry import AABB, CameraCalibration, project, ray_dirs_camera, zdepth_to_raydist
from livefire.render.blackbody import T_LUT_MAX, T_LUT_MIN, blackbody_lut
from livefire.render.overlay import OverlayMap
from livefire.render.volume import march_kernel
from livefire.scene.snapshot import SceneSnapshot, assert_calib_version

wp.config.quiet = True

SOOT_ACTIVE_EPS = 1e-6  # soot above this counts as an active cell (absorption below is negligible)


@dataclass
class ArrayFireState:
    """Simple ``FireStateView`` backed by NumPy or Warp arrays (prescribed fields; tests and demo).

    T_g, soot: (G,G,G) float32 cell-centred [ix,iy,iz] (K, soot density kg/m^3); char, removed: (N_e,)
    float32 0..1; h_gas metres; roi world AABB; flame: optional (G,G,G) float32 W/m^3 (None = no flame)."""

    T_g: Any
    soot: Any
    char: Any
    removed: Any
    h_gas: float
    roi: AABB
    flame: Any = None


def _to_np(a: Any) -> NDArray:
    return a.numpy() if isinstance(a, wp.array) else np.asarray(a)


def _to_wp3(a: Any, device: str) -> wp.array:
    if isinstance(a, wp.array) and a.dtype == wp.float32 and a.ndim == 3 and str(a.device) == device:
        return a
    return wp.array(np.ascontiguousarray(_to_np(a), dtype=np.float32), dtype=wp.float32, device=device)


def _srgb_to_linear_lut() -> NDArray[np.float32]:
    x = np.arange(256, dtype=np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb_u8(lin: NDArray) -> NDArray[np.uint8]:
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.ascontiguousarray(np.rint(s * 255.0).astype(np.uint8))


class _View:
    """Static per-camera data (volume-resolution rays, occlusion limits, overlay map)."""

    def __init__(self, snap: SceneSnapshot, calib: CameraCalibration, scale: float, p: Params, device: str) -> None:
        W, H = calib.width, calib.height
        self.calib = calib
        self.scale = scale
        self.Ws, self.Hs = max(1, round(W * scale)), max(1, round(H * scale))
        K = calib.K.copy()
        K[:2, :2] *= scale
        K[0, 2] = (calib.K[0, 2] + 0.5) * scale - 0.5
        K[1, 2] = (calib.K[1, 2] + 0.5) * scale - 0.5
        self.calib_s = CameraCalibration(calib.cam_id, self.Ws, self.Hs, K, calib.dist, calib.T_world_from_camera,
                                         calib.calib_version)  # fmt: skip
        dc = ray_dirs_camera(self.calib_s)
        nrm = np.linalg.norm(dc, axis=-1, keepdims=True)
        R = calib.T_world_from_camera[:3, :3]
        self.dirs_np = ((dc / nrm) @ R.T).astype(np.float32)
        self.origin = calib.center_world
        self.dirs = wp.array(self.dirs_np, dtype=wp.vec3, device=device)
        # nearest full-res pixel for each volume pixel (occlusion depth / hole mask)
        iu = np.clip(np.rint((np.arange(self.Ws) + 0.5) / scale - 0.5).astype(int), 0, W - 1)
        iv = np.clip(np.rint((np.arange(self.Hs) + 0.5) / scale - 0.5).astype(int), 0, H - 1)
        depth = snap.view_depth[calib.cam_id][np.ix_(iv, iu)].astype(np.float64)
        hole = snap.view_hole_mask[calib.cam_id][np.ix_(iv, iu)] | (depth <= 0.0)
        t_occ = zdepth_to_raydist(self.calib_s, depth) - p["feather_m"]
        o = np.broadcast_to(self.origin, self.dirs_np.reshape(-1, 3).shape)
        tn, tf, hit = snap.roi.ray_intersect(o, self.dirs_np.reshape(-1, 3).astype(np.float64))
        self.t_near = tn.reshape(self.Hs, self.Ws)
        self.t_far = np.minimum(np.where(hit, tf, 0.0).reshape(self.Hs, self.Ws), t_occ)
        self.t_far = np.where(hole, 0.0, self.t_far)  # hole pixels march nothing
        self.overlay = OverlayMap(snap, calib, snap.view_depth[calib.cam_id], p["depth_tol"])

    def pixel_rect(self, box: AABB) -> tuple[int, int, int, int]:
        """Conservative (u0,u1,v0,v1) exclusive-end volume-pixel rectangle covering a world box, clamped to
        the image; empty (u1<=u0 or v1<=v0) when the box projects fully outside the view."""
        c = np.array([[x, y, z] for x in (box.min[0], box.max[0]) for y in (box.min[1], box.max[1])
                      for z in (box.min[2], box.max[2])])  # fmt: skip
        uv, z, ok = project(self.calib_s, c)
        if not ok.all() or z.min() < 0.05:
            return 0, self.Ws, 0, self.Hs
        u0 = min(max(int(np.floor(uv[:, 0].min())) - 2, 0), self.Ws)
        u1 = min(max(int(np.ceil(uv[:, 0].max())) + 3, 0), self.Ws)
        v0 = min(max(int(np.floor(uv[:, 1].min())) - 2, 0), self.Hs)
        v1 = min(max(int(np.ceil(uv[:, 1].max())) + 3, 0), self.Hs)
        return u0, u1, v0, v1  # empty (u1<=u0 or v1<=v0) when the box is entirely off-view


class Compositor:
    """``RendererBackend`` implementation (Warp kernels on ``device``). Not thread-safe."""

    def __init__(self, snapshot: SceneSnapshot, params: Params | None = None, device: str | None = None,
                 volume_scale: float | None = None) -> None:  # fmt: skip
        self.p = params or load_params("render")
        self.t_amb = float(load_params("sim")["T_amb"])  # ambient for the flame visual temperature
        self.device = device or str(load_yaml("app").get("device", "cpu"))
        self.snap = snapshot
        self.scale = float(volume_scale if volume_scale is not None else self.p["volume_scale"])
        self.lut = wp.array(blackbody_lut(), dtype=wp.vec3, device=self.device)
        self.views = {cid: _View(snapshot, c, self.scale, self.p, self.device) for cid, c in snapshot.cameras.items()}
        self._srgb = _srgb_to_linear_lut()
        self.ash = np.asarray(self.p.vec("ash_color"), dtype=np.float32)
        self._latency_ms = 0.0
        self.last_volume_ms = 0.0
        self.last_march_ms = 0.0

    # --- RendererBackend metrics
    @property
    def latency_ms(self) -> float:
        """Latency of the last ``render`` call, ms (Warp synchronised)."""
        return self._latency_ms

    @property
    def peak_vram_allocated_mb(self) -> float:
        return 0.0  # CPU: no VRAM accounting (GPU accounting is added with the GPU run)

    @property
    def peak_vram_reserved_mb(self) -> float:
        return 0.0

    def _active_box(self, T: NDArray, soot: NDArray, fire: Any, flame: NDArray | None = None) -> AABB | None:
        """World AABB of cells that can emit/absorb (hot, sooty or flaming: ``flame > 0.01 flame_ref``), padded
        by one cell for trilinear support."""
        act = (T > self.p["T_emit_min"]) | (soot > SOOT_ACTIVE_EPS)
        if flame is not None:
            act = act | (flame > 0.01 * self.p["flame_ref"])
        if not act.any():
            return None
        h = float(fire.h_gas)
        g = T.shape[0]
        lo = np.array([np.flatnonzero(act.any(axis=tuple(a for a in range(3) if a != ax)))[[0, -1]]
                       for ax in range(3)])  # fmt: skip
        i0 = np.clip(lo[:, 0] - 1, 0, g)
        i1 = np.clip(lo[:, 1] + 2, 0, g)  # cell index + 1 is the far edge; +1 more cell of support
        rmin = np.asarray(fire.roi.min)
        return AABB(rmin + i0 * h, rmin + np.maximum(i1, i0 + 1) * h)

    def render_volume(self, cam_id: int, fire: Any) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Volume only: ``(T_vol (H,W), L_vol (H,W,3))`` float32 at full frame resolution, linear light."""
        view = self.views[cam_id]
        t_start = time.perf_counter()
        Tn, Sn = _to_np(fire.T_g), _to_np(fire.soot)
        g = Tn.shape[0]
        h = float(fire.h_gas)
        Hs, Ws = view.Hs, view.Ws
        T_out = np.ones((Hs, Ws), np.float32)
        L_out = np.zeros((Hs, Ws, 3), np.float32)
        flame = getattr(fire, "flame", None)
        Fn = None if flame is None else _to_np(flame)
        box = self._active_box(Tn, Sn, fire, Fn)
        rect = view.pixel_rect(box) if box is not None else None
        if box is not None and rect is not None and rect[1] > rect[0] and rect[3] > rect[2]:
            u0, u1, v0, v1 = rect
            dsub = view.dirs_np[v0:v1, u0:u1].reshape(-1, 3).astype(np.float64)
            o = np.broadcast_to(view.origin, dsub.shape)
            an, af, ahit = box.ray_intersect(o, dsub)
            sh = (v1 - v0, u1 - u0)
            t0 = np.full((Hs, Ws), 0.0, np.float32)
            t1 = np.full((Hs, Ws), 0.0, np.float32)
            t0[v0:v1, u0:u1] = np.maximum(view.t_near[v0:v1, u0:u1], an.reshape(sh))
            t1[v0:v1, u0:u1] = np.minimum(view.t_far[v0:v1, u0:u1], np.where(ahit, af, 0.0).reshape(sh))
            dev = self.device
            t_k = time.perf_counter()
            Tw, Sw = _to_wp3(fire.T_g, dev), _to_wp3(fire.soot, dev)
            Fw = wp.zeros((g, g, g), dtype=wp.float32, device=dev) if flame is None else _to_wp3(flame, dev)
            oT = wp.array(T_out, dtype=wp.float32, device=dev)
            oL = wp.array(L_out, dtype=wp.vec3, device=dev)
            p = self.p
            wp.launch(
                march_kernel, dim=(v1 - v0, u1 - u0),
                inputs=[Tw, Sw, Fw, self.lut, view.dirs, wp.array(t0, dtype=wp.float32, device=dev),
                        wp.array(t1, dtype=wp.float32, device=dev), wp.vec3(*view.origin), wp.vec3(*fire.roi.min),
                        h, g, p["step_factor"] * h, p["t_min"], p["emission_scale"], p["T_emit_min"],
                        p["T_emit_ref"], p["emission_power"], p["k_abs"], self.t_amb, p["dT_flame"], p["flame_ref"],
                        p["soot_albedo"] * p["ambient"],
                        T_LUT_MIN, T_LUT_MAX, u0, v0, oT, oL],
                device=dev,
            )  # fmt: skip
            wp.synchronize_device(dev)
            self.last_march_ms = (time.perf_counter() - t_k) * 1e3
            T_out, L_out = oT.numpy(), oL.numpy()
        if (Hs, Ws) != (view.calib.height, view.calib.width):
            size = (view.calib.width, view.calib.height)
            T_out = cv2.resize(T_out, size, interpolation=cv2.INTER_LINEAR)
            L_out = cv2.resize(L_out, size, interpolation=cv2.INTER_LINEAR)
        self.last_volume_ms = (time.perf_counter() - t_start) * 1e3
        return T_out, L_out

    def render(self, cam_id: int, frame: Frame, fire: Any) -> NDArray[np.uint8]:
        """Composite the fire over ``frame`` for camera ``cam_id``; see module docstring.

        Raises KeyError for an unknown cam_id and CalibVersionError for a calib mismatch."""
        if cam_id not in self.views:
            raise KeyError(f"unknown cam_id {cam_id}")
        assert_calib_version(self.snap, frame)
        view = self.views[cam_id]
        if frame.rgb.shape[:2] != (view.calib.height, view.calib.width):
            raise ValueError(f"frame shape {frame.rgb.shape} != camera {view.calib.height}x{view.calib.width}")
        t0 = time.perf_counter()
        T_vol, L_vol = self.render_volume(cam_id, fire)
        p = self.p
        base = self._srgb[frame.rgb]  # (H,W,3) linear
        char = view.overlay.splat(_to_np(fire.char))
        removed = view.overlay.splat(_to_np(fire.removed))
        base = base * (1.0 - p["char_dark"] * char)[..., None]
        base = base + (self.ash - base) * removed[..., None]
        Le = p["exposure"] * L_vol
        out = base * T_vol[..., None] + Le / (1.0 + Le)
        res = _linear_to_srgb_u8(out)
        self._latency_ms = (time.perf_counter() - t0) * 1e3
        return res


def blob_state(snap: SceneSnapshot, G: int, points: NDArray, peak_K: float = 1600.0, sigma_cells: float = 2.0,
               T_amb: float = 293.15) -> ArrayFireState:  # fmt: skip
    """Prescribed fire state: Gaussian hot blob(s) (peak ``peak_K``, sigma ``sigma_cells`` h) on a T_amb
    field, no soot, no char. ``points`` (M,3) world metres. G cells per side over ``snap.roi``."""
    h = float(snap.roi.size[0] / G)
    ax = [snap.roi.min[i] + (np.arange(G) + 0.5) * h for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    T = np.full((G, G, G), T_amb)
    for pt in np.atleast_2d(points):
        r2 = (X - pt[0]) ** 2 + (Y - pt[1]) ** 2 + (Z - pt[2]) ** 2
        T = np.maximum(T, T_amb + (peak_K - T_amb) * np.exp(-r2 / (2 * (sigma_cells * h) ** 2)))
    n = len(snap.pos)
    return ArrayFireState(T.astype(np.float32), np.zeros((G, G, G), np.float32), np.zeros(n, np.float32),
                          np.zeros(n, np.float32), h, snap.roi)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    """CLI: ``--demo`` renders the synthetic table scene with a prescribed blob and a charred/ash patch
    and writes ``cam0_composite.png`` and ``cam1_composite.png`` to ``--out``."""
    from livefire.scene.builder import snapshot_from_synthetic
    from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--G", type=int, default=48)
    args = ap.parse_args(argv)
    if not args.demo:
        ap.error("only --demo is implemented")
    snap = snapshot_from_synthetic()
    comp = Compositor(snap)
    scene, cams = build_table_v1(), load_cameras()
    st = blob_state(snap, args.G, np.array([[-0.075, 0.0, 0.2]]))
    # prescribed char/ash on the wood-block and paper elements (demo only)
    wood = snap.obj_id == 1
    st.char[wood] = 0.8
    st.removed[snap.obj_id == 3] = 0.6
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid, calib in cams.items():
        v = render_view(scene, calib, seed=0)
        res = comp.render(cid, Frame(cid, v.rgb, 0, 0, calib.calib_version), st)
        cv2.imwrite(str(out / f"cam{cid}_composite.png"), res[..., ::-1])
        print(f"cam{cid}: render {comp.latency_ms:.0f} ms (volume {comp.last_volume_ms:.0f} ms, march "
              f"{comp.last_march_ms:.0f} ms), wrote {out / f'cam{cid}_composite.png'}  [{device_label(comp.device)}]")


if __name__ == "__main__":
    main()
