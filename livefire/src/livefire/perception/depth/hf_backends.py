"""Hugging Face ``transformers`` depth backends (optional ``depth`` extra: torch + transformers).

Candidates (ADR-0005): Depth Anything V2 Small, metric-indoor fine-tune (``kind='metric'``) and relative
(``kind='disparity'``). Weights are cached under ``data/models`` (``configs/depth.yaml: model_cache_dir``). On CUDA
the network runs in fp16; output is float32 host NumPy resized to the input resolution. ``torch``/``transformers``
are imported lazily so ``import livefire`` works without the extra.
"""

from __future__ import annotations

import numpy as np

from livefire.common.errors import DepthError
from livefire.perception.depth.base import (
    Kind,
    MetricsMixin,
    RawDepth,
    default_device,
    load_depth_config,
    model_cache_dir,
    require_infer_ready,
)
from livefire.perception.types import DepthResult


def depth_extra_available() -> bool:
    """True when torch and transformers can be imported."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class HFDepthBackend(MetricsMixin):
    """``DepthBackend`` for an HF ``AutoModelForDepthEstimation`` checkpoint.

    Input ``Frame`` (rgb uint8 (H,W,3) undistorted, host); the backend copies to the device. ``infer_raw`` returns
    the un-anchored ``RawDepth`` at (H,W); ``infer`` returns a ``DepthResult`` for metric models and raises
    ``DepthError`` for disparity models (wrap them in ``AnchoredDepthBackend``). ``conf`` of ``infer`` is a constant
    0.5 on valid pixels (the models give no confidence; uncalibrated heuristic). Thread-safe (serialised).
    """

    def __init__(
        self,
        hf_id: str,
        kind: Kind,
        *,
        device: str | None = None,
        fp16_on_cuda: bool = True,
        input_size: int = 518,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise DepthError("the 'depth' extra is not installed (uv sync --extra depth)") from e
        self.hf_id, self.kind = hf_id, kind
        self.device = device or default_device()
        self._torch = torch
        self._half = fp16_on_cuda and self.device.startswith("cuda")
        cache = str(cache_dir or model_cache_dir())
        try:
            self._proc = AutoImageProcessor.from_pretrained(hf_id, cache_dir=cache)
            if input_size > 0:
                self._proc.size = {"height": input_size, "width": input_size}
            model = AutoModelForDepthEstimation.from_pretrained(hf_id, cache_dir=cache)
        except Exception as e:
            raise DepthError(f"could not load depth model {hf_id}: {e}") from e
        model.eval().to(self.device)
        self._model = model.half() if self._half else model

    def warmup(self, shape: tuple[int, int]) -> None:
        """Run one inference on a black (H, W) image (allocator and kernel warm-up)."""
        self._infer_np(np.zeros((shape[0], shape[1], 3), np.uint8))

    def _infer_np(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        with self._lock, torch.inference_mode():
            inp = self._proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
            if self._half:
                inp = inp.half()
            out = self._model(pixel_values=inp).predicted_depth  # (1, h', w')
            out = torch.nn.functional.interpolate(out[:, None].float(), size=(h, w), mode="bilinear",
                                                  align_corners=False)[0, 0]
            return out.cpu().numpy().astype(np.float32)

    def infer_raw(self, frame) -> RawDepth:
        """Un-anchored depth/disparity at input resolution (host NumPy)."""
        try:
            vals = self._timed(lambda: self._infer_np(np.ascontiguousarray(frame.rgb)))
        except DepthError:
            raise
        except Exception as e:
            raise DepthError(f"{self.hf_id} inference failed: {e}") from e
        valid = np.isfinite(vals)
        if self.kind == "metric":
            valid &= vals > 0.0
        return RawDepth(np.where(valid, vals, 0.0).astype(np.float32), valid, self.kind, frame.calib_version)

    def infer(self, frame) -> DepthResult:
        """Metric ``DepthResult`` (``DepthError`` for disparity models)."""
        return require_infer_ready(self.infer_raw(frame))


def create_depth_backend(name: str | None = None, cfg: dict | None = None, device: str | None = None):
    """Backend from ``configs/depth.yaml`` (``name`` overrides ``backend``). ``oracle`` is not built here (it needs a
    calibration and scene): construct ``OracleDepthBackend`` directly."""
    cfg = cfg or load_depth_config()
    name = name or cfg["backend"]
    if name not in cfg["backends"]:
        raise DepthError(f"unknown depth backend {name!r}; configured: {sorted(cfg['backends'])}")
    b = cfg["backends"][name]
    return HFDepthBackend(
        b["hf_id"], b["kind"], device=device, fp16_on_cuda=bool(cfg["fp16_on_cuda"]),
        input_size=int(cfg["input_size"]),
    )
