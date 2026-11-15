"""Acquisition bundle for the Gate 1 evaluation tools (P1-007).

A live snapshot is not stored anywhere (``live`` cannot be replayed), so the evaluation tools run one acquisition
themselves (calibration -> cameras -> depth -> fusion, the same ``acquire_with_report`` as live mode) and keep what
they need in an :class:`AcqBundle`: per camera the frozen **view depth** (float32 (H,W) Z-depth, metres, 0 = hole) and
hole mask, the last *undistorted* RGB frame, the calibration (undistorted ``K``, world pose; world = P1R2 table frame,
metres, +Z up), the ChArUco board spec, the fusion ROI and the acquisition report. The bundle round-trips through one
``.npz`` file (no pickle) so the user can acquire once and run every evaluation offline, and send the file back.

Host NumPy only; ``snapshot()`` builds a surfel-free ``SceneSnapshot`` that is good enough for click back-projection
(``app.click``), which only reads ``cameras``, ``view_depth``, ``view_hole_mask`` and ``surfel_spacing``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from livefire.calib.board import BoardSpec
from livefire.common.config import load_yaml
from livefire.common.geometry import AABB, CameraCalibration
from livefire.scene.snapshot import ObjectRegistry, SceneSnapshot

FORMAT_VERSION = 1


@dataclass(frozen=True)
class AcqBundle:
    """Frozen acquisition products. ``view_depth[c]`` f32 (H,W) m (0 = hole), ``view_hole[c]`` bool (H,W),
    ``rgb[c]`` uint8 (H,W,3) undistorted RGB (may be empty), all in the undistorted pixel space of
    ``cameras[c]``. ``roi`` is the fusion ROI of ``configs/acquisition.yaml`` (world m): the volume that was reconstructed and is judged; ``board`` the calibration board; ``label`` says where the data
    came from (e.g. ``real`` or ``synthetic table_v1, oracle depth``); ``report`` is the acquisition report."""

    cameras: dict[int, CameraCalibration]
    view_depth: dict[int, np.ndarray]
    view_hole: dict[int, np.ndarray]
    rgb: dict[int, np.ndarray]
    roi: AABB
    surfel_spacing: float
    board: BoardSpec | None = None
    label: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    created: str = ""

    def snapshot(self) -> SceneSnapshot:
        """Surfel-free read-only snapshot over the stored views (for ``app.click`` back-projection)."""
        cal_versions = {c.calib_version for c in self.cameras.values()}
        return SceneSnapshot(
            version=1, calib_version=cal_versions.pop() if len(cal_versions) == 1 else 0, roi=self.roi,
            h_gas=float(self.roi.size.max()) / 32.0, cameras=dict(self.cameras),
            pos=np.zeros((0, 3), np.float32), normal=np.zeros((0, 3), np.float32), area=np.zeros(0, np.float32),
            obj_id=np.zeros(0, np.int32), material_id=np.zeros(0, np.uint8), material_conf=np.zeros(0, np.float32),
            neighbors=np.zeros((0, 1), np.int32), solid_mask=np.zeros((2, 2, 2), np.uint8),
            solid_fraction=np.zeros((2, 2, 2), np.float32),
            view_depth={c: d.astype(np.float32, copy=False) for c, d in self.view_depth.items()},
            view_hole_mask={c: h.astype(np.bool_, copy=False) for c, h in self.view_hole.items()},
            objects=ObjectRegistry(), surfel_spacing=self.surfel_spacing,
        )  # fmt: skip

    # ---- persistence ---------------------------------------------------------------------------------------------
    def save(self, path: str | Path, *, write_pngs: bool = True) -> Path:
        """Write ``path`` (``.npz``) and, if ``write_pngs``, ``view_cam<c>.png`` (undistorted frame) and
        ``depth_cam<c>.png`` (colour-mapped view depth) next to it. Returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": FORMAT_VERSION, "label": self.label, "created": self.created, "report": self.report,
            "surfel_spacing": self.surfel_spacing, "roi": {"min": self.roi.min.tolist(), "max": self.roi.max.tolist()},
            "board": None if self.board is None else self.board.to_dict(),
            "cameras": [{"cam_id": c.cam_id, "width": c.width, "height": c.height, "K": c.K.tolist(),
                         "dist": c.dist.tolist(), "T_world_from_camera": c.T_world_from_camera.tolist(),
                         "calib_version": c.calib_version} for c in self.cameras.values()],
        }  # fmt: skip
        arrays: dict[str, Any] = {"meta": np.array(json.dumps(meta, default=_json_default))}
        for c in self.cameras:
            arrays[f"depth_{c}"] = self.view_depth[c].astype(np.float32)
            arrays[f"hole_{c}"] = self.view_hole[c].astype(np.bool_)
            if c in self.rgb:
                arrays[f"rgb_{c}"] = self.rgb[c].astype(np.uint8)
        np.savez_compressed(p, **arrays)
        if write_pngs:
            for c, img in self.rgb.items():
                cv2.imwrite(str(p.with_name(f"view_cam{c}.png")), np.ascontiguousarray(img[..., ::-1]))
            for c, d in self.view_depth.items():
                cv2.imwrite(str(p.with_name(f"depth_cam{c}.png")), _depth_png(d, self.view_hole[c]))
        return p

    @staticmethod
    def load(path: str | Path) -> AcqBundle:
        """Read a bundle written by :meth:`save`. Raises ``ValueError`` for an unknown format."""
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported bundle format {meta.get('format')!r}")
            cams = {int(c["cam_id"]): CameraCalibration(int(c["cam_id"]), int(c["width"]), int(c["height"]),
                                                        np.array(c["K"]), np.array(c["dist"]),
                                                        np.array(c["T_world_from_camera"]), int(c["calib_version"]))
                    for c in meta["cameras"]}  # fmt: skip
            depth = {c: np.array(z[f"depth_{c}"]) for c in cams}
            hole = {c: np.array(z[f"hole_{c}"]) for c in cams}
            rgb = {c: np.array(z[f"rgb_{c}"]) for c in cams if f"rgb_{c}" in z.files}
        return AcqBundle(cams, depth, hole, rgb, AABB(np.array(meta["roi"]["min"]), np.array(meta["roi"]["max"])),
                         float(meta["surfel_spacing"]), None if meta["board"] is None else BoardSpec.from_dict(meta["board"]),
                         str(meta["label"]), dict(meta["report"]), str(meta["created"]))  # fmt: skip


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _depth_png(depth: np.ndarray, hole: np.ndarray) -> np.ndarray:
    ok = ~hole & (depth > 0)
    img = np.zeros(depth.shape, np.uint8)
    if ok.any():
        lo, hi = float(depth[ok].min()), float(depth[ok].max())
        img[ok] = np.clip(255 * (depth[ok] - lo) / max(hi - lo, 1e-6), 1, 255).astype(np.uint8)
    out = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    out[~ok] = 0
    return out


# ---- acquisition ------------------------------------------------------------------------------------------------
def bundle_from_snapshot(snap: SceneSnapshot, rgb: Mapping[int, np.ndarray], report: dict[str, Any], board: BoardSpec | None,
                         label: str, roi: AABB | None = None) -> AcqBundle:
    """Bundle from an acquired ``SceneSnapshot`` plus the undistorted frames ``rgb[c]`` (uint8 RGB). ``roi``: the region
    the evaluations judge (default: the snapshot's simulation ROI; ``acquire_bundle`` passes the fusion ROI)."""
    return AcqBundle(dict(snap.cameras), {c: np.array(d) for c, d in snap.view_depth.items()},
                     {c: np.array(h) for c, h in snap.view_hole_mask.items()}, {c: np.array(v) for c, v in rgb.items()},
                     roi or snap.roi, float(snap.surfel_spacing), board, label, report, time.strftime("%Y-%m-%dT%H:%M:%S"))  # fmt: skip


def acquire_bundle(sources: Mapping[int, Any], calibration: Any, depth_factory: Callable[[dict[int, CameraCalibration]], Any],
                   *, grid_cfg: dict[str, Any] | None = None, cfg: dict[str, Any] | None = None,
                   progress: Callable[[str], None] | None = None, label: str = "real") -> AcqBundle:
    """Run one acquisition on started ``sources`` (``{cam_id: CameraSource}``) and return the bundle.

    ``calibration``: ``calib.store.CalibrationSet`` (frames are undistorted with the stored lens model) or
    ``{cam_id: CameraCalibration}`` (frames already undistorted). ``depth_factory``: ``live.make_depth_factory(...)``
    (called once with the calibrations; the models are dropped on return). The last undistorted frame of every camera
    after the acquisition is stored for display and landmark detection. Raises ``ReconstructionError`` /
    ``DepthError`` from acquisition."""
    from livefire.scene.acquire import acquire_with_report, prepare_frame

    recs = getattr(calibration, "cameras", None)
    if recs is not None and all(hasattr(r, "calibration") for r in recs.values()):
        cals = {c: r.calibration for c, r in recs.items()}
        und = {c: r.undistorter() for c, r in recs.items()}
        board = calibration.board
    else:
        cals, und, board = dict(calibration), {}, None
    from livefire.fusion.objects import load_acquisition_config

    cfg = cfg or load_acquisition_config()
    res = acquire_with_report(sources, calibration, depth_factory(cals), cfg, grid_cfg, progress=progress)
    fr = cfg["fusion_roi"]
    roi = AABB(np.asarray(fr["min"], float), np.asarray(fr["max"], float))
    rgb = {}
    for c, cal in cals.items():
        f = sources[c].latest()
        if f is not None:
            rgb[c] = np.array(prepare_frame(f, cal, und.get(c)).rgb)
    return bundle_from_snapshot(res.snapshot, rgb, res.report, board, label, roi)


def acquire_synthetic_bundle(depth: str = "oracle", grid: int = 32, seconds: float = 0.3, max_frames: int = 6,
                             progress: Callable[[str], None] | None = None) -> AcqBundle:
    """# MOCK-style demo/test input: acquire the synthetic ``table_v1`` two-camera rig. ``depth`` is a
    ``configs/depth.yaml`` backend or ``oracle`` (ground-truth depth anchored to the table plane). Fast
    (CPU, a few seconds) with the defaults."""
    from livefire.app.live import make_depth_factory
    from livefire.capture.synthetic_source import SyntheticCameraSource
    from livefire.fusion.objects import load_acquisition_config
    from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

    scene, cams = build_table_v1(), load_cameras()
    srcs = {c: SyntheticCameraSource(c, render_view(scene, cams[c]).rgb, cams[c].calib_version, fps=200) for c in cams}
    acq = load_acquisition_config()
    acq.update(seconds=seconds, max_frames=max_frames)
    for s in srcs.values():
        s.start()
    try:
        return acquire_bundle(srcs, cams, make_depth_factory(depth), grid_cfg={**load_yaml("grid"), "G": grid}, cfg=acq,
                              progress=progress, label=f"synthetic table_v1, {depth} depth, CPU")
    finally:
        for s in srcs.values():
            s.stop()


# ---- shared command-line plumbing ---------------------------------------------------------------------------------
def add_source_args(ap: argparse.ArgumentParser) -> None:
    """Options shared by every evaluation tool: where the acquisition comes from."""
    g = ap.add_argument_group("acquisition (use --bundle to reuse an earlier one)")
    g.add_argument("--bundle", default=None, help="an acquisition.npz from an earlier run (skips the cameras)")
    g.add_argument("--synthetic", action="store_true", help="use the synthetic table_v1 rig instead of cameras (demo/test)")
    g.add_argument("--cameras", default=None, help="cameras file (default configs/cameras.yaml)")
    g.add_argument("--calib", default="active", help="calibration version or 'active'")
    g.add_argument("--calib-root", default="data/calib")
    g.add_argument("--depth", default=None, help="depth backend (default configs/depth.yaml; 'oracle' = synthetic only)")
    g.add_argument("--grid", type=int, default=None, help="gas grid G override (acquisition only needs the ROI)")
    g.add_argument("--out", default="out/eval", help="where a new acquisition bundle is saved (out/eval/acq_<time>/)")


def resolve_bundle(a: argparse.Namespace, say: Callable[[str], None] = print) -> AcqBundle:
    """Bundle for the parsed ``add_source_args`` options: load ``--bundle``, or acquire from the synthetic rig or
    from the real cameras (and save the result under ``--out``). Raises ``SystemExit`` with guidance on user errors."""
    if a.bundle:
        say(f"loading acquisition bundle {a.bundle}")
        return AcqBundle.load(a.bundle)
    if a.synthetic:
        say("acquiring the synthetic table_v1 rig (demo data, not your scene)")
        b = acquire_synthetic_bundle(a.depth or "oracle", a.grid or 32, progress=None)
    else:
        b = _acquire_real(a, say)
    path = b.save(Path(a.out) / time.strftime("acq_%Y%m%d_%H%M%S") / "acquisition.npz")
    say(f"saved acquisition bundle {path} (reuse with --bundle {path})")
    return b


def _acquire_real(a: argparse.Namespace, say: Callable[[str], None]) -> AcqBundle:
    from livefire.app.live import make_depth_factory
    from livefire.calib.store import load_calibration
    from livefire.capture.factory import load_camera_configs, make_source

    calibration = load_calibration(Path(a.calib_root), None if a.calib == "active" else int(a.calib))
    cfgs = [c for c in load_camera_configs(a.cameras) if int(c["cam_id"]) in calibration.cameras]
    missing = set(calibration.cameras) - {int(c["cam_id"]) for c in cfgs}
    if missing:
        raise SystemExit(f"calibration v{calibration.version} has cameras {sorted(missing)} that are not in the cameras file")
    sources = {int(c["cam_id"]): make_source(c, calib_version=0) for c in cfgs}
    app_cfg = load_yaml("app")
    grid_cfg = {**load_yaml("grid"), **({"G": a.grid} if a.grid else {})}
    say("acquiring: do not touch the scene or the cameras for about 10 s (more on a CPU)")
    for s in sources.values():
        s.start()
    try:
        return acquire_bundle(sources, calibration, make_depth_factory(a.depth, str(app_cfg.get("device", "cpu"))),
                              grid_cfg=grid_cfg, progress=lambda m: say("  " + m), label=f"real, calib v{calibration.version}")
    finally:
        for s in sources.values():
            s.stop()
