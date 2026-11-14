"""Shared pieces of the depth backends: raw output type, metrics, device and config helpers.

Frames: camera OpenCV (+Z forward), depth is Z-depth in metres, invalid = 0.0 (PRD 4). Backends run on the device
selected by ``configs/app.yaml: device``; ``RawDepth`` and ``DepthResult`` live in host NumPy memory and are
owned by the caller.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from livefire.common.config import load_yaml
from livefire.common.errors import DepthError
from livefire.perception.types import DepthResult

Kind = Literal["metric", "disparity"]


@dataclass(frozen=True, eq=False)
class RawDepth:
    """Un-anchored network output in the model's native space, host memory, (H, W) at the input resolution.

    ``kind='metric'``: ``values`` f32 metres (arbitrary scale error, depth-like).
    ``kind='disparity'``: ``values`` f32 affine-invariant disparity, larger = nearer (``d = a / z + b``).
    ``valid`` bool (H, W) marks finite model output. ``calib_version`` is that of the input frame.
    """

    values: np.ndarray
    valid: np.ndarray
    kind: Kind
    calib_version: int

    def __post_init__(self) -> None:
        if self.values.dtype != np.float32 or self.values.ndim != 2:
            raise ValueError(f"values must be float32 (H,W), got {self.values.dtype} {self.values.shape}")
        if self.valid.dtype != np.bool_ or self.valid.shape != self.values.shape:
            raise ValueError("valid must be bool with the shape of values")
        if not np.isfinite(self.values[self.valid]).all():
            raise ValueError("valid pixels must be finite")


def depth_result_from_z(z: np.ndarray, valid: np.ndarray, conf: np.ndarray | float, calib_version: int) -> DepthResult:
    """Build a contract-clean ``DepthResult``: non-finite or non-positive depth becomes invalid (0.0)."""
    z = np.asarray(z, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(z) & (z > 0.0)
    zc = np.ascontiguousarray(np.where(ok, z, 0.0), dtype=np.float32)
    c = np.where(ok, np.broadcast_to(np.asarray(conf, dtype=np.float32), z.shape), 0.0)
    return DepthResult(zc, ok, np.ascontiguousarray(np.clip(c, 0.0, 1.0), dtype=np.float32), int(calib_version))


class MetricsMixin:
    """``latency_ms`` and peak VRAM properties of the ``DepthBackend`` protocol (shared timer semantics).

    Subclasses call ``_timed(fn)``; on CUDA the device is synchronised before and after and the peak memory
    counters are read from ``torch.cuda``. On CPU the VRAM properties are 0.0.
    """

    device: str = "cpu"

    def __init__(self) -> None:
        self._lat_ms = 0.0
        self._lock = threading.Lock()

    @property
    def latency_ms(self) -> float:
        """Latency of the last ``infer`` call in ms."""
        return self._lat_ms

    @property
    def peak_vram_allocated_mb(self) -> float:
        """Peak allocated CUDA memory in MB (0.0 on CPU)."""
        if not str(self.device).startswith("cuda"):
            return 0.0
        import torch

        return float(torch.cuda.max_memory_allocated(self.device)) / 1e6

    @property
    def peak_vram_reserved_mb(self) -> float:
        """Peak reserved CUDA memory in MB (0.0 on CPU)."""
        if not str(self.device).startswith("cuda"):
            return 0.0
        import torch

        return float(torch.cuda.max_memory_reserved(self.device)) / 1e6

    def reset_peak_vram(self) -> None:
        """Reset the CUDA peak counters (no-op on CPU)."""
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats(self.device)

    def _sync(self) -> None:
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)

    def _timed(self, fn):
        self._sync()
        t0 = time.perf_counter_ns()
        out = fn()
        self._sync()
        self._lat_ms = (time.perf_counter_ns() - t0) / 1e6
        return out


def load_depth_config() -> dict:
    """``configs/depth.yaml`` as a dict."""
    return load_yaml("depth")


def default_device() -> str:
    """Device string from ``configs/app.yaml: device``."""
    return str(load_yaml("app").get("device", "cpu"))


def model_cache_dir(cfg: dict | None = None) -> Path:
    """Weight cache directory (``data/models`` by default), created on demand; pathlib only."""
    cfg = cfg or load_depth_config()
    p = Path(cfg.get("model_cache_dir", "data/models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_infer_ready(raw: RawDepth) -> DepthResult:
    """Convert a *metric* ``RawDepth`` to ``DepthResult`` as-is; disparity needs anchoring first."""
    if raw.kind != "metric":
        raise DepthError("relative (disparity) backend output must be anchored to the table plane; wrap the "
                         "backend in AnchoredDepthBackend")
    return depth_result_from_z(raw.values, raw.valid, 0.5, raw.calib_version)
