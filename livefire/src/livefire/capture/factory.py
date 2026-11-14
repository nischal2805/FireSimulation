"""Camera source factory driven by ``configs/cameras.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from livefire.capture.file_source import VideoFileSource
from livefire.capture.opencv_source import OpenCVCameraSource
from livefire.capture.synthetic_source import SyntheticCameraSource
from livefire.common.config import ConfigError, load_yaml, load_yaml_text
from livefire.common.interfaces import CameraSource

TYPES = ("usb", "url", "file", "synthetic")


def load_camera_configs(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Camera entries from ``configs/cameras.yaml`` (or ``path``); validates cam_id uniqueness and ``type``."""
    data = load_yaml("cameras") if path is None else load_yaml_text(Path(path).read_text())
    cams = (data or {}).get("cameras")
    if not isinstance(cams, list) or not cams:
        raise ConfigError("cameras.yaml: 'cameras' must be a non-empty list")
    ids = [c.get("cam_id") for c in cams]
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) for i in ids):
        raise ConfigError(f"cameras.yaml: cam_id must be unique integers, got {ids}")
    for c in cams:
        if c.get("type") not in TYPES:
            raise ConfigError(f"cameras.yaml: cam {c['cam_id']}: type must be one of {TYPES}, got {c.get('type')!r}")
    return cams


def make_source(cfg: dict[str, Any], calib_version: int = 0) -> CameraSource:
    """Build (not start) a ``CameraSource`` from one camera entry ``{cam_id, type, source, width, height,
    fps, backend}``. ``calib_version`` is the active calibration's version (0 = uncalibrated)."""
    cam_id, typ = int(cfg["cam_id"]), cfg.get("type")
    w, h, fps = int(cfg.get("width", 0)), int(cfg.get("height", 0)), float(cfg.get("fps", 0))
    if typ == "usb":
        return OpenCVCameraSource(cam_id, int(cfg["source"]), w, h, fps, cfg.get("backend", "auto"), calib_version)
    if typ == "url":
        return OpenCVCameraSource(cam_id, str(cfg["source"]), w, h, fps, cfg.get("backend", "auto"), calib_version)
    if typ == "file":
        return VideoFileSource(cam_id, cfg["source"], bool(cfg.get("realtime", True)), bool(cfg.get("loop", False)), fps, calib_version)
    if typ == "synthetic":
        from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

        cams = load_cameras()
        rgb = render_view(build_table_v1(), cams[cam_id % len(cams)], seed=0).rgb
        return SyntheticCameraSource(cam_id, rgb, calib_version, fps or 30.0)
    raise ConfigError(f"unknown camera type {typ!r}; expected one of {TYPES}")
