"""Segmentation backend factory (``configs/segment.yaml``). Imports without torch; the real backends are imported lazily."""

from __future__ import annotations

from livefire.common.errors import SegmentationError
from livefire.perception.segment.base import load_segment_config

BACKEND_NAMES = ("oracle", "gdino_sam2", "sam3", "mask2former")


def create_segmentation_backend(name: str | None = None, cfg: dict | None = None, device: str | None = None, *,
                                calib=None, scene=None):
    """Backend named ``name`` (default ``configs/segment.yaml: backend``).

    ``oracle`` (# MOCK:) needs the camera ``calib`` (and optionally a ``scene``); the model backends need the
    ``perception`` extra and cached or downloadable weights. ``SegmentationError`` for an unknown name, a missing
    extra, a missing ``calib`` or a model that cannot be loaded.
    """
    cfg = cfg or load_segment_config()
    name = name or cfg["backend"]
    if name not in BACKEND_NAMES:
        raise SegmentationError(f"unknown segmentation backend {name!r}; choose one of {list(BACKEND_NAMES)}")
    if name == "oracle":
        if calib is None:
            raise SegmentationError("the oracle backend needs calib=<CameraCalibration> (it renders the synthetic scene)")
        from livefire.perception.segment.oracle import OracleSegmentationBackend

        return OracleSegmentationBackend(calib, scene, cfg=cfg)
    from livefire.perception.segment import hf_backends

    cls = {"gdino_sam2": hf_backends.GDinoSam2Backend, "sam3": hf_backends.Sam3Backend,
           "mask2former": hf_backends.Mask2FormerBackend}[name]
    return cls(cfg, device)
