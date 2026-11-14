"""Table-plane depth anchoring (P1R3, PRD 14 "scale anchor").

Monocular depth has an unknown scale (metric models: residual scale error) or an unknown scale and shift
(relative models: affine-invariant disparity). The calibrated table plane ``z = 0`` (world frame, +Z up) is seen in
every view; for pixels inside the table region the analytic Z-depth of that plane is known exactly from the
calibration. We fit the model's native-space mapping to it with RANSAC + least squares and apply it to the whole
frame:

* ``metric``:    ``z = scale * raw``                               (raw in metres, f32 (H,W))
* ``disparity``: ``raw = a / z + b``  =>  ``z = a / (raw - b)``     (scale ``a`` > 0, shift ``b``)

Frames: ``CameraCalibration`` poses are ``T_world_from_camera``; depth is Z-depth along camera +Z in metres;
invalid = 0.0. All arrays are host NumPy; functions return new arrays. Raises ``DepthError`` when the fit is not
trustworthy (too few table pixels, inlier ratio below the configured minimum, non-positive scale, degenerate
plane geometry).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from livefire.common.errors import CalibVersionError, DepthError
from livefire.common.geometry import CameraCalibration, ray_dirs_camera
from livefire.perception.depth.base import (
    MetricsMixin,
    RawDepth,
    depth_result_from_z,
    load_depth_config,
)
from livefire.perception.types import DepthResult


def _val(x):
    """Unwrap a PRD 8.1 ``{value, unit, range, status}`` entry; plain scalars pass through."""
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass(frozen=True)
class AnchorConfig:
    """Anchor tunables (``configs/depth.yaml: anchor``)."""

    min_inlier_ratio: float = 0.5
    min_pixels: int = 400
    ransac_iters: int = 300
    inlier_thresh_rel: float = 0.03
    z_max_m: float = 20.0
    min_disparity_span_rel: float = 0.05  # table 1/z span (p95-p5) / mean below this: shift is ill-conditioned
    seed: int = 1234
    max_fit_points: int = 20000  # RANSAC subsample; the final least-squares refit uses all inliers

    @classmethod
    def from_yaml(cls, cfg: dict | None = None) -> AnchorConfig:
        """Build from ``configs/depth.yaml`` (or a given dict with the same layout)."""
        a = (cfg or load_depth_config())["anchor"]
        return cls(
            min_inlier_ratio=float(a["min_inlier_ratio"]),
            min_pixels=int(a["min_pixels"]),
            ransac_iters=int(a["ransac_iters"]),
            inlier_thresh_rel=float(a["inlier_thresh_rel"]),
            z_max_m=float(_val(a["z_max_m"])),
            min_disparity_span_rel=float(_val(a["min_disparity_span_rel"])),
            seed=int(a["seed"]),
            max_fit_points=int(_val(a["max_fit_points"])),
        )


@dataclass(frozen=True)
class TableRegion:
    """Table region as a polygon of world (x, y) vertices in metres on the plane ``z = 0``."""

    polygon_xy: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.polygon_xy, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
            raise ValueError(f"polygon_xy must be (N>=3, 2), got {p.shape}")
        object.__setattr__(self, "polygon_xy", p)

    @classmethod
    def from_footprint(cls, half_extent_xy: tuple[float, float], margin_m: float = 0.0) -> TableRegion:
        """Axis-aligned rectangle centred on the world origin (board centre), expanded by ``margin_m``."""
        hx, hy = half_extent_xy[0] + margin_m, half_extent_xy[1] + margin_m
        return cls(np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]))

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> TableRegion:
        """Region from ``configs/depth.yaml: anchor.region``."""
        r = (cfg or load_depth_config())["anchor"]["region"]
        if r["mode"] == "polygon_xy":
            return cls(np.asarray(r["polygon_xy"], dtype=np.float64))
        if r["mode"] == "board_footprint":
            return cls.from_footprint(tuple(r["board_half_extent_xy"]), float(_val(r["margin_m"])))
        raise DepthError(f"unknown anchor region mode {r['mode']!r}")


@dataclass(frozen=True)
class AnchorFit:
    """Result of an anchor fit. ``kind`` metric: ``z = scale * raw`` (``shift`` = 0); disparity:
    ``raw = scale / z + shift``. Residuals are Z-depth errors in metres between the anchored depth and the analytic
    plane depth on table pixels: ``resid_*`` over RANSAC inliers, ``resid_mae_all_m`` over all table pixels."""

    kind: str
    scale: float
    shift: float
    inlier_ratio: float
    n_pixels: int
    n_inliers: int
    resid_mae_m: float
    resid_rms_m: float
    resid_p95_m: float
    resid_mae_all_m: float


def table_plane_depth(calib: CameraCalibration, region: TableRegion) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Z-depth of the plane ``z = 0`` per pixel, and the mask of pixels whose ray hits it inside ``region``.

    Returns ``(plane_z f64 (H,W) metres, 0.0 outside the mask; mask bool (H,W))``. Z-depth equals the ray parameter
    ``t`` because camera rays are ``inv(K) @ [u,v,1]`` (z component 1). Rays that do not hit the plane in front of
    the camera are excluded.
    """
    R = calib.T_world_from_camera[:3, :3]
    c = calib.center_world
    d = ray_dirs_camera(calib) @ R.T  # (H,W,3) world-frame directions
    dz = d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -c[2] / dz
    hit = np.isfinite(t) & (t > 0.0)
    t = np.where(hit, t, 0.0)
    x = c[0] + t * d[..., 0]
    y = c[1] + t * d[..., 1]
    inside = MplPath(region.polygon_xy).contains_points(np.stack([x.ravel(), y.ravel()], axis=1)).reshape(t.shape)
    mask = hit & inside
    return np.where(mask, t, 0.0), mask


def _lsq_metric(x: np.ndarray, zp: np.ndarray) -> float:
    """Scale minimising sum(((s x - zp) / zp)^2) (relative residuals)."""
    r = x / zp
    return float((r @ np.ones_like(r)) / (r @ r))


def _lsq_disparity(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    A = np.stack([w, np.ones_like(w)], axis=1)
    sol, *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(sol[0]), float(sol[1])


def fit_anchor(
    raw: np.ndarray,
    raw_valid: np.ndarray,
    plane_z: np.ndarray,
    mask: np.ndarray,
    kind: str,
    cfg: AnchorConfig | None = None,
) -> AnchorFit:
    """Fit the native-space mapping on table pixels (RANSAC then least squares on the inliers).

    ``raw`` f32 (H,W) model output (metres for ``metric``, disparity for ``disparity``); ``plane_z`` / ``mask`` from
    ``table_plane_depth``. RANSAC is seeded (``cfg.seed``). Raises ``DepthError`` below ``cfg.min_pixels`` table
    pixels, below ``cfg.min_inlier_ratio``, for a non-positive scale, or for a table that spans too little depth
    range to fit a shift.
    """
    cfg = cfg or AnchorConfig()
    if kind not in ("metric", "disparity"):
        raise DepthError(f"unknown depth kind {kind!r}")
    sel = mask & raw_valid & np.isfinite(raw)
    if kind == "metric":
        sel &= raw > 0.0
    x_all = raw[sel].astype(np.float64)
    zp_all = plane_z[sel].astype(np.float64)
    n = int(x_all.size)
    if n < cfg.min_pixels:
        raise DepthError(f"only {n} usable table pixels (< {cfg.min_pixels}); table region not visible enough")
    rng = np.random.default_rng(cfg.seed)
    sub = rng.choice(n, size=min(n, cfg.max_fit_points), replace=False) if n > cfg.max_fit_points else np.arange(n)
    xs, zs = x_all[sub], zp_all[sub]
    m = xs.size

    if kind == "metric":
        pick = rng.integers(0, m, cfg.ransac_iters)
        s_hyp = zs[pick] / xs[pick]
        s_hyp = np.concatenate([s_hyp, [_lsq_metric(xs, zs)]])
        # inliers per hypothesis: |s x - zp| <= tau zp
        counts = np.array([(np.abs(s * xs - zs) <= cfg.inlier_thresh_rel * zs).sum() for s in s_hyp])
        s0 = s_hyp[int(np.argmax(counts))]
        scale, shift = s0, 0.0
        for _ in range(3):  # LSQ refit on inliers, re-evaluate
            inl = np.abs(scale * x_all - zp_all) <= cfg.inlier_thresh_rel * zp_all
            if inl.sum() < 1:
                break
            scale = _lsq_metric(x_all[inl], zp_all[inl])
        inl = np.abs(scale * x_all - zp_all) <= cfg.inlier_thresh_rel * zp_all
        z_est = scale * x_all
    else:
        w_all = 1.0 / zp_all
        span_w = np.percentile(w_all, 95) - np.percentile(w_all, 5)
        if span_w < cfg.min_disparity_span_rel * float(np.mean(w_all)):
            raise DepthError("table region spans too little depth range to fit scale and shift "
                             f"(1/z span {span_w:.4f} 1/m)")
        ws = 1.0 / zs
        tau = cfg.inlier_thresh_rel * (np.percentile(x_all, 95) - np.percentile(x_all, 5))
        i0, i1 = rng.integers(0, m, cfg.ransac_iters), rng.integers(0, m, cfg.ransac_iters)
        dw = ws[i1] - ws[i0]
        ok = np.abs(dw) > 0.25 * span_w
        i0, i1, dw = i0[ok], i1[ok], dw[ok]
        if i0.size == 0:
            raise DepthError("RANSAC found no well-conditioned sample pair on the table region")
        a_h = (xs[i1] - xs[i0]) / dw
        b_h = xs[i0] - a_h * ws[i0]
        best, best_cnt = (a_h[0], b_h[0]), -1
        for a, b in zip(a_h, b_h, strict=True):
            cnt = int((np.abs(a * ws + b - xs) <= tau).sum())
            if cnt > best_cnt:
                best, best_cnt = (float(a), float(b)), cnt
        scale, shift = best
        for _ in range(3):
            inl = np.abs(scale * w_all + shift - x_all) <= tau
            if inl.sum() < 3:
                break
            scale, shift = _lsq_disparity(x_all[inl], w_all[inl])
        inl = np.abs(scale * w_all + shift - x_all) <= tau
        denom = x_all - shift
        with np.errstate(divide="ignore", invalid="ignore"):
            z_est = np.where(denom > 0, scale / denom, np.inf)

    if not np.isfinite(scale) or scale <= 0.0:
        raise DepthError(f"anchor fit gave a non-positive scale ({scale:.4g}); depth model unusable on this view")
    ratio = float(inl.sum()) / n
    if ratio < cfg.min_inlier_ratio:
        raise DepthError(f"anchor inlier ratio {ratio:.3f} below minimum {cfg.min_inlier_ratio:.3f}")
    err = np.minimum(np.abs(z_est - zp_all), 1e6)  # clip once: inf (denominator <= 0) must not reach any resid_* stat
    e_in = err[inl]
    return AnchorFit(
        kind=kind, scale=float(scale), shift=float(shift), inlier_ratio=ratio, n_pixels=n, n_inliers=int(inl.sum()),
        resid_mae_m=float(e_in.mean()), resid_rms_m=float(np.sqrt((e_in**2).mean())),
        resid_p95_m=float(np.percentile(e_in, 95)),
        resid_mae_all_m=float(err.mean()),
    )  # fmt: skip


def apply_anchor(raw: RawDepth, fit: AnchorFit, z_max_m: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Map the raw output to Z-depth metres: ``(z f32 (H,W), valid bool (H,W))``; invalid pixels hold 0.0."""
    x = raw.values.astype(np.float64)
    if fit.kind == "metric":
        z = fit.scale * x
        ok = raw.valid & (z > 0.0) & (z <= z_max_m)
    else:
        denom = x - fit.shift
        ok = raw.valid & (denom > fit.scale / z_max_m)
        z = np.where(ok, fit.scale / np.where(ok, denom, 1.0), 0.0)
    ok &= np.isfinite(z)
    return np.where(ok, z, 0.0).astype(np.float32), ok


class AnchoredDepthBackend(MetricsMixin):
    """``DepthBackend`` that anchors a raw backend (``infer_raw``) to the calibrated table plane each call.

    Input: ``Frame`` of the calibrated camera (``calib_version`` must equal ``calib.calib_version``, else
    ``CalibVersionError``). Output ``DepthResult`` (host NumPy, Z-depth m, invalid 0.0); ``conf`` is the table-fit
    inlier ratio on valid pixels (an uncalibrated heuristic, not a probability). ``last_fit`` holds the latest
    ``AnchorFit``. Latency covers network and anchoring. Thread-safe (calls are serialised).
    """

    def __init__(self, backend, calib: CameraCalibration, region: TableRegion, cfg: AnchorConfig | None = None) -> None:
        super().__init__()
        self.backend = backend
        self.calib = calib
        self.region = region
        self.cfg = cfg or AnchorConfig.from_yaml()
        self.device = getattr(backend, "device", "cpu")
        self.last_fit: AnchorFit | None = None
        self._plane_z, self._mask = table_plane_depth(calib, region)

    @property
    def peak_vram_allocated_mb(self) -> float:
        return self.backend.peak_vram_allocated_mb

    @property
    def peak_vram_reserved_mb(self) -> float:
        return self.backend.peak_vram_reserved_mb

    def warmup(self, shape: tuple[int, int]) -> None:
        """Warm up the wrapped backend for an (H, W) input."""
        self.backend.warmup(shape)

    def infer(self, frame) -> DepthResult:
        """Anchor and return metric depth; ``DepthError`` if the fit is rejected."""
        if frame.calib_version != self.calib.calib_version:
            raise CalibVersionError(f"frame calib_version {frame.calib_version} != {self.calib.calib_version}")
        with self._lock:
            t0 = time.perf_counter_ns()
            raw = self.backend.infer_raw(frame)
            if raw.values.shape != (self.calib.height, self.calib.width):
                raise DepthError(f"depth shape {raw.values.shape} != calibration {(self.calib.height, self.calib.width)}")
            fit = fit_anchor(raw.values, raw.valid, self._plane_z, self._mask, raw.kind, self.cfg)
            z, valid = apply_anchor(raw, fit, self.cfg.z_max_m)
            self.last_fit = fit
            self._lat_ms = (time.perf_counter_ns() - t0) / 1e6
            return depth_result_from_z(z, valid, fit.inlier_ratio, raw.calib_version)
