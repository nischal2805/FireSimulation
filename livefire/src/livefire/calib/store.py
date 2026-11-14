"""Versioned calibration store: ``data/calib/calib_vNNN.yaml`` (+ ``calib_vNNN/`` images and manifest), ``active`` pointer.

Per camera the YAML holds the fitted ``K_raw`` (distorted image), ``dist`` (5), the *undistorted* ``K`` (alpha policy of
``intrinsics``), resolution and ``T_world_from_camera``. Loaders return ``livefire.common.geometry.CameraCalibration``
whose ``K`` is the undistorted K, so frames must be undistorted with ``Undistorter`` (same K) before geometry use.
Undistort policy: ``cv2.initUndistortRectifyMap(K_raw, dist, None, K_undist, (w, h))``, alpha 0 = no invalid border
pixels, same output size. Paths use pathlib only.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from livefire.calib.board import BoardSpec
from livefire.calib.extrinsics import ExtrinsicsResult
from livefire.calib.intrinsics import IntrinsicsResult
from livefire.common.config import load_yaml_text
from livefire.common.geometry import CameraCalibration

DEFAULT_ROOT = Path("data") / "calib"
_VER = re.compile(r"^calib_v(\d{3,})\.yaml$")


class CalibrationError(ValueError):
    """Calibration missing, malformed, or inconsistent with the frames it is applied to."""


def _mat(a: NDArray) -> list:
    return np.asarray(a, np.float64).tolist()


def list_versions(root: Path) -> list[int]:
    """Stored calib_version numbers, ascending."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(int(m.group(1)) for p in root.iterdir() if (m := _VER.match(p.name)))


def next_version(root: Path) -> int:
    v = list_versions(root)
    return (v[-1] + 1) if v else 1


def calib_path(root: Path, version: int) -> Path:
    return Path(root) / f"calib_v{version:03d}.yaml"


def active_version(root: Path) -> int | None:
    """Version named by the ``active`` pointer file, or None."""
    p = Path(root) / "active"
    if not p.is_file():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError as e:
        raise CalibrationError(f"{p}: not a version number") from e


def set_active(root: Path, version: int) -> None:
    if version not in list_versions(root):
        raise CalibrationError(f"calibration version {version} does not exist in {root}")
    (Path(root) / "active").write_text(f"{version}\n")


# ---------------------------------------------------------------------------- intrinsics (per camera, pre-extrinsics)


def _intr_to_dict(r: IntrinsicsResult) -> dict:
    return {"cam_id": r.cam_id, "width": r.width, "height": r.height, "K_raw": _mat(r.K_raw), "dist": _mat(r.dist),
            "K": _mat(r.K_undist), "alpha": r.alpha,
            "validation": {"rms_fit_px": r.rms_fit, "median_fit_px": r.median_fit, "rms_heldout_px": r.rms_heldout,
                           "median_heldout_px": r.median_heldout, "n_fit": r.n_fit, "n_heldout": r.n_heldout},
            "fit_images": r.fit_names, "heldout_images": r.heldout_names}


def _intr_from_dict(d: dict) -> IntrinsicsResult:
    v = d["validation"]
    return IntrinsicsResult(int(d["cam_id"]), int(d["width"]), int(d["height"]), np.array(d["K_raw"], float),
                            np.array(d["dist"], float), np.array(d["K"], float), float(d["alpha"]), v["rms_fit_px"],
                            v["rms_heldout_px"], v["median_heldout_px"], v["median_fit_px"], v["n_fit"], v["n_heldout"],
                            list(d.get("fit_images", [])), list(d.get("heldout_images", [])))


def save_intrinsics(root: Path, res: IntrinsicsResult, spec: BoardSpec, images: dict[str, NDArray] | None = None) -> Path:
    """Write ``<root>/intrinsics/cam<N>/intrinsics.yaml`` (+ ``images/``); returns the yaml path."""
    d = Path(root) / "intrinsics" / f"cam{res.cam_id}"
    (d / "images").mkdir(parents=True, exist_ok=True)
    for name, img in (images or {}).items():
        cv2.imwrite(str(d / "images" / name), img)
    p = d / "intrinsics.yaml"
    p.write_text(yaml.safe_dump({**_intr_to_dict(res), "board": spec.to_dict()}, sort_keys=False))
    return p


def load_intrinsics(root: Path, cam_id: int) -> tuple[IntrinsicsResult, BoardSpec]:
    p = Path(root) / "intrinsics" / f"cam{cam_id}" / "intrinsics.yaml"
    if not p.is_file():
        raise CalibrationError(f"no intrinsics for cam{cam_id} at {p}: run `calib intrinsics --cam {cam_id}` first")
    d = load_yaml_text(p.read_text())
    return _intr_from_dict(d), BoardSpec.from_dict(d["board"])


# ---------------------------------------------------------------------------- full calibration


@dataclass(frozen=True)
class CameraRecord:
    """Loaded camera: ``calibration`` (undistorted K, world pose) plus ``K_raw`` for undistortion."""

    calibration: CameraCalibration
    K_raw: NDArray[np.float64]
    alpha: float

    def undistorter(self) -> Undistorter:
        return Undistorter(self)


@dataclass(frozen=True)
class CalibrationSet:
    version: int
    board: BoardSpec
    cameras: dict[int, CameraRecord]
    manifest: dict


class Undistorter:
    """Undistorts frames of exactly the calibrated resolution to the stored undistorted ``K`` (maps built once)."""

    def __init__(self, rec: CameraRecord) -> None:
        c = rec.calibration
        self.width, self.height, self.cam_id = c.width, c.height, c.cam_id
        self._map1, self._map2 = cv2.initUndistortRectifyMap(rec.K_raw, c.dist, None, c.K, (c.width, c.height), cv2.CV_32FC1)

    def apply(self, frame: NDArray) -> NDArray:
        """Frame (H,W[,C]) uint8 at the calibrated resolution -> undistorted frame, same shape. Raises on mismatch."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise CalibrationError(f"cam{self.cam_id}: calibration is for {self.width}x{self.height}, frame is "
                                   f"{frame.shape[1]}x{frame.shape[0]}; recalibrate or change the capture resolution")
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)


def check_frame_size(cal: CameraCalibration, frame_hw: tuple[int, int]) -> None:
    """Raise CalibrationError if (H, W) differs from the calibration's resolution."""
    if (frame_hw[1], frame_hw[0]) != (cal.width, cal.height):
        raise CalibrationError(f"cam{cal.cam_id}: calibration is {cal.width}x{cal.height}, frame is "
                               f"{frame_hw[1]}x{frame_hw[0]}")


def save_calibration(
    root: Path,
    spec: BoardSpec,
    intrinsics: dict[int, IntrinsicsResult],
    ext: ExtrinsicsResult,
    images: dict[int, NDArray] | None = None,
    *,
    make_active: bool = True,
) -> int:
    """Write the next ``calib_vNNN.yaml`` (+ ``calib_vNNN/`` images + manifest); returns the new version number."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ver = next_version(root)
    cams = []
    ext_by = {c.cam_id: c for c in ext.cameras}
    for cid, c in sorted(ext_by.items()):
        d = _intr_to_dict(intrinsics[cid])
        d["T_world_from_camera"] = _mat(c.T_world_from_camera)
        d["extrinsics"] = {"reproj_rms_px": c.reproj_rms_px, "n_corners": c.n_corners, "flipped": c.flipped}
        cams.append(d)
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc = {"calib_version": ver, "created": created, "board": spec.to_dict(), "world_frame": "P1R2: origin board centre, +Z toward cameras",
           "cameras": cams,
           "consistency_m": {f"{a}-{b}": d for (a, b), d in ext.consistency_m.items()}, "warnings": list(ext.warnings)}
    folder = root / f"calib_v{ver:03d}"
    folder.mkdir(exist_ok=True)
    manifest = {"calib_version": ver, "created": created, "yaml": calib_path(root, ver).name, "images": [], "cameras": sorted(ext_by)}
    for cid, img in (images or {}).items():
        name = f"images/extrinsics_cam{cid}.png"
        (folder / "images").mkdir(exist_ok=True)
        cv2.imwrite(str(folder / name), img)
        manifest["images"].append(name)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    calib_path(root, ver).write_text(yaml.safe_dump(doc, sort_keys=False))
    if make_active:
        set_active(root, ver)
    return ver


def load_calibration(root: Path = DEFAULT_ROOT, version: int | None = None) -> CalibrationSet:
    """Load ``version`` (default: the ``active`` pointer, else the newest). Cameras keyed by cam_id."""
    root = Path(root)
    if version is None:
        version = active_version(root)
    if version is None:
        vs = list_versions(root)
        if not vs:
            raise CalibrationError(f"no calibration in {root}: run `calib extrinsics` (see README)")
        version = vs[-1]
    p = calib_path(root, version)
    if not p.is_file():
        raise CalibrationError(f"calibration file not found: {p}")
    d = load_yaml_text(p.read_text())
    cams = {}
    for c in d["cameras"]:
        cal = CameraCalibration(int(c["cam_id"]), int(c["width"]), int(c["height"]), np.array(c["K"], float),
                                np.array(c["dist"], float), np.array(c["T_world_from_camera"], float), int(d["calib_version"]))
        cams[cal.cam_id] = CameraRecord(cal, np.array(c["K_raw"], float), float(c["alpha"]))
    mf = root / f"calib_v{version:03d}" / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.is_file() else {}
    return CalibrationSet(int(d["calib_version"]), BoardSpec.from_dict(d["board"]), cams, manifest)
