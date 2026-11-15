"""Scene acquisition (PRD 6.1, 7.2; P1R3, P1R4): frames -> anchored depth -> median accumulation -> TSDF fusion ->
surface extraction -> table plane + object clustering -> ``SceneSnapshot``.

Frames: world (metres, +Z up, table plane z = 0), camera OpenCV. Everything is host NumPy except the TSDF volume, which
lives on the Warp device of ``configs/app.yaml`` for the duration of the call. Works with one or two (or more) cameras.
The returned snapshot owns its arrays; the sources are started/stopped by the caller.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from livefire.capture.types import Frame
from livefire.common.config import load_yaml
from livefire.common.errors import CalibVersionError, ReconstructionError
from livefire.common.geometry import AABB, CameraCalibration, project
from livefire.common.log import get_logger
from livefire.fusion.accumulate import AccumulatedDepth, accumulate_depth
from livefire.fusion.extract import extract_surface
from livefire.fusion.objects import (
    DROPPED,
    TABLE_ID,
    cluster_objects,
    fit_table_plane,
    load_acquisition_config,
)
from livefire.fusion.tsdf import TSDFVolume
from livefire.perception.material.types import MATERIAL_CLASSES, MaterialPrediction
from livefire.scene.builder import ViewInput, manual_material, snapshot_from_surfels
from livefire.scene.snapshot import SceneSnapshot
from livefire.scene.surfels import ViewPoints, bin_surfels

log = get_logger(__name__)


def _val(x: Any) -> Any:
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass(frozen=True, eq=False)
class AcquisitionResult:
    """``snapshot`` plus ``report`` (JSON-friendly dict: frames, anchor fits, plane, objects, timings, warnings) and
    ``timings_ms`` (stage -> ms, one sample per stage; ``total`` is the whole acquisition)."""

    snapshot: SceneSnapshot
    report: dict[str, Any]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _cal_of(calibration: Any) -> tuple[dict[int, CameraCalibration], dict[int, Any]]:
    """``CalibrationSet`` (with undistorters) or ``{cam_id: CameraCalibration}`` (frames already undistorted)."""
    cams = getattr(calibration, "cameras", None)
    if cams is not None and all(hasattr(r, "calibration") for r in cams.values()):
        return {c: r.calibration for c, r in cams.items()}, {c: r.undistorter() for c, r in cams.items()}
    log.warning("calibration given without undistorters: frames are assumed to be already undistorted")
    return dict(calibration), {}


def _backends(depth_backend: Any, cals: dict[int, CameraCalibration]) -> dict[int, Any]:
    """Per-camera backends: a ``{cam_id: backend}`` mapping, a factory ``calib -> backend`` or one shared backend."""
    if isinstance(depth_backend, Mapping):
        return {c: depth_backend[c] for c in cals}
    if hasattr(depth_backend, "infer"):
        return {c: depth_backend for c in cals}
    return {c: depth_backend(cals[c]) for c in cals}


def collect_frames(
    sources: Mapping[int, Any] | list[Any],
    cals: dict[int, CameraCalibration],
    undistorters: dict[int, Any],
    seconds: float,
    max_frames: int,
    min_frames: int,
    first_frame_timeout_s: float = 10.0,
) -> dict[int, list[Frame]]:
    """Sample each started source over ``seconds``: at most ``max_frames`` new frames per camera, spaced
    ``seconds / max_frames`` apart. Frames are undistorted (``Undistorter``) unless the source already stamped the
    calibration's ``calib_version`` on them, and re-stamped with it. A source reporting status ``ended`` stops
    contributing. Raises ``ReconstructionError`` if a camera yields fewer than ``min_frames``."""
    srcs = dict(sources) if isinstance(sources, Mapping) else {s.cam_id: s for s in sources}
    missing = set(cals) - set(srcs)
    if missing:
        raise ReconstructionError(f"no source for calibrated cameras {sorted(missing)}")
    srcs = {c: srcs[c] for c in cals}
    period = seconds / max_frames
    out: dict[int, list[Frame]] = {c: [] for c in srcs}
    last_seq = {c: -1 for c in srcs}
    nxt = dict.fromkeys(srcs, 0.0)
    t0 = time.perf_counter()
    t_end = t0 + seconds
    while True:
        now = time.perf_counter()
        active = False
        for c, s in srcs.items():
            if len(out[c]) >= max_frames:
                continue
            f = s.latest()
            ended = getattr(s, "status", "") == "ended"
            if ended and f is not None and f.seq == last_seq[c]:
                continue
            active = True
            if f is None or f.seq == last_seq[c] or now - t0 < nxt[c]:
                continue
            ver = cals[c].calib_version
            if f.calib_version not in (0, ver):
                raise CalibVersionError(f"camera {c}: frame calib_version {f.calib_version} != calibration {ver}")
            rgb = f.rgb
            if f.calib_version != ver and c in undistorters:
                rgb = undistorters[c].apply(rgb)
            out[c].append(Frame(c, rgb, f.t_arrive_ns, f.seq, ver))
            last_seq[c] = f.seq
            nxt[c] = (now - t0) + period
        if not active or now >= t_end and all(len(v) >= min_frames for v in out.values()):
            break
        if now >= t_end + first_frame_timeout_s:
            break
        time.sleep(0.002)
    for c, v in out.items():
        if len(v) < min_frames:
            raise ReconstructionError(f"camera {c}: only {len(v)} frames in the acquisition window (need {min_frames})")
    return out


def _observes(calib: CameraCalibration, pos: np.ndarray, acc: AccumulatedDepth, tol: float) -> int:
    """Number of element positions whose projected Z-depth agrees with the accumulated depth (within ``tol`` m)."""
    uv, z, ok = project(calib, pos)
    u, v = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok &= (u >= 0) & (u < calib.width) & (v >= 0) & (v < calib.height)
    m = acc.depth[np.where(ok, v, 0), np.where(ok, u, 0)]
    return int((ok & acc.valid[np.where(ok, v, 0), np.where(ok, u, 0)] & (np.abs(m - z) <= tol)).sum())


def acquire_with_report(
    sources: Mapping[int, Any] | list[Any],
    calibration: Any,
    depth_backend: Any,
    cfg: dict[str, Any] | None = None,
    grid_cfg: dict[str, Any] | None = None,
    version: int = 1,
    device: str | None = None,
) -> AcquisitionResult:
    """Acquire a frozen scene. ``sources``: started ``CameraSource`` objects (list or ``{cam_id: source}``);
    ``calibration``: ``calib.store.CalibrationSet`` or ``{cam_id: CameraCalibration}``; ``depth_backend``: a
    ``{cam_id: DepthBackend}`` mapping, a factory ``CameraCalibration -> DepthBackend`` or one shared backend; each
    camera's backend must already anchor depth to the table plane (``AnchoredDepthBackend``, P1R3) and return
    metric Z-depth. ``cfg`` = ``configs/acquisition.yaml: acquisition`` and ``grid_cfg`` = ``configs/grid.yaml``
    (defaults loaded). Raises ``ReconstructionError`` if too few frames / no surface is found."""
    cfg = cfg or load_acquisition_config()
    grid_cfg = grid_cfg or load_yaml("grid")
    cals, undists = _cal_of(calibration)
    backends = _backends(depth_backend, cals)
    timings: dict[str, float] = {}

    @contextmanager
    def stage(name: str):  # accumulates wall-clock ms per stage (stages that run per camera are summed)
        t = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t) * 1e3

    t_all = time.perf_counter()
    warnings: list[str] = []
    report: dict[str, Any] = {"cameras": {}, "warnings": warnings}

    with stage("capture"):
        frames = collect_frames(sources, cals, undists, float(_val(cfg["seconds"])), int(cfg["max_frames"]),
                                int(cfg["min_frames"]))  # fmt: skip

    acc: dict[int, AccumulatedDepth] = {}
    for c, fl in frames.items():
        scales: list[float] = []
        zs = np.zeros((len(fl), cals[c].height, cals[c].width), dtype=np.float32)
        vs = np.zeros(zs.shape, dtype=bool)
        with stage("depth"):
            for i, f in enumerate(fl):
                r = backends[c].infer(f)
                zs[i], vs[i] = r.z_depth_m, r.valid
                fit = getattr(backends[c], "last_fit", None)
                if fit is not None:
                    scales.append(fit.scale)
        with stage("accumulate"):
            acc[c] = accumulate_depth(zs, vs, float(_val(cfg["agree_tol_rel"])), float(_val(cfg["min_confidence"])))
        del zs, vs
        report["cameras"][c] = {
            "frames": len(fl),
            "valid_fraction": float(acc[c].valid.mean()),
            "mean_conf": float(acc[c].conf[acc[c].valid].mean()) if acc[c].valid.any() else 0.0,
            "anchor_scale_median": float(np.median(scales)) if scales else None,
            "anchor_scale_spread": float(np.ptp(scales)) if scales else None,
        }

    fr = cfg["fusion_roi"]
    roi_f = AABB(np.asarray(fr["min"], float), np.asarray(fr["max"], float))
    h = float(_val(cfg["voxel_size"]))
    with stage("tsdf"):
        vol = TSDFVolume(roi_f, h, float(cfg["truncation_voxels"]), device)
        for c in cals:
            vol.integrate(cals[c], acc[c].depth, acc[c].conf, float(_val(cfg["max_depth_m"])))
        tsdf, wgt = vol.tsdf(), vol.weight()
    with stage("extract"):
        sp = extract_surface(tsdf, wgt, vol.grid)
    if len(sp.pos) < 10:
        raise ReconstructionError(f"fusion produced only {len(sp.pos)} surface samples (anchoring or ROI wrong?)")
    s = float(grid_cfg["surfel_spacing"])
    with stage("objects"):
        plane = fit_table_plane(sp.pos, sp.normal, sp.footprint, cfg)
        if not plane.agrees:
            warnings.append(f"table plane deviates {1e3 * plane.z_dev_m:.1f} mm from z = 0")
        lab = cluster_objects(sp.pos, sp.normal, sp.footprint, plane, cfg, s)
        keep = lab != DROPPED
    with stage("snapshot"):
        roi_g = AABB(np.asarray(grid_cfg["roi_min"], float), np.asarray(grid_cfg["roi_max"], float))
        vp = ViewPoints(sp.pos[keep], sp.normal[keep], lab[keep], sp.footprint[keep], np.ones(int(keep.sum()), bool))
        sf = bin_surfels([vp], roi_g, s, min_points=1)
        ids = sorted(set(np.unique(sf.obj_id).tolist()) | {TABLE_ID})
        mats: dict[int, MaterialPrediction] = {}
        names: dict[int, str] = {}
        dm = str(cfg["default_object_material"])
        for o in ids:
            if o == TABLE_ID:
                mats[o], names[o] = _prior("other_nonflammable", 1.0, False), "table"
            else:
                mats[o], names[o] = _prior(dm, float(cfg["default_object_confidence"]), True), f"object_{o}"
        views = {
            c: ViewInput(cals[c], acc[c].depth, acc[c].valid, np.full(acc[c].depth.shape, -1, np.int32))
            for c in cals
        }
        obs: dict[int, dict[int, int]] = {}
        min_obs = int(cfg["min_view_observations"])
        for o in ids:
            p = sf.pos[sf.obj_id == o].astype(np.float64)
            obs[o] = {c: o for c in cals if _observes(cals[c], p, acc[c], max(2 * h, 0.02)) >= min(min_obs, len(p)) and len(p) > 0}
        snap = snapshot_from_surfels(sf, views, mats, grid_cfg, version, names, observations=obs)

    total_ms = (time.perf_counter() - t_all) * 1e3
    timings["total"] = total_ms
    report.update(
        {
            "voxel_size_m": h,
            "tsdf_shape": list(vol.grid.shape),
            "device": vol.device,
            "n_samples": len(sp.pos),
            "n_dropped_samples": int((~keep).sum()),
            "n_elements": len(snap.pos),
            "plane": {"z_dev_mm": 1e3 * plane.z_dev_m, "tilt_deg": plane.tilt_deg, "agrees": plane.agrees,
                      "n_inliers": plane.n_inliers},
            "objects": {
                int(o): {"name": names[o], "elements": int((snap.obj_id == o).sum()),
                         "area_m2": float(snap.area[snap.obj_id == o].sum()), "cameras": sorted(obs[o]),
                         "material": MATERIAL_CLASSES[snap.objects[o].material.material_id],
                         "requires_review": snap.objects[o].material.requires_review}
                for o in ids
            },
            "timings_ms": timings,
            "acquisition_s": total_ms / 1e3,
        }
    )
    return AcquisitionResult(snap, report, timings)


def _prior(material: str, conf: float, review: bool) -> MaterialPrediction:
    """One-hot category-prior material (source ``category_prior``)."""
    m = manual_material(material)
    return MaterialPrediction(m.class_probabilities, None, conf, "category_prior", review)


def acquire_snapshot(
    sources: Mapping[int, Any] | list[Any],
    calibration: Any,
    depth_backend: Any,
    cfg: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SceneSnapshot:
    """``acquire_with_report(...).snapshot`` (see there for arguments)."""
    return acquire_with_report(sources, calibration, depth_backend, cfg, **kwargs).snapshot


__all__ = ["AcquisitionResult", "acquire_snapshot", "acquire_with_report", "collect_frames"]
