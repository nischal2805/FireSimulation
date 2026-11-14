"""Oracle depth backend: exact depth from the synthetic renderer (tests and sanity benchmarks only)."""

from __future__ import annotations

import numpy as np

from livefire.common.errors import CalibVersionError, DepthError
from livefire.common.geometry import CameraCalibration
from livefire.perception.depth.base import Kind, MetricsMixin, RawDepth, require_infer_ready
from livefire.perception.types import DepthResult
from livefire.scene.synthetic import SyntheticScene, build_table_v1, render_view


class OracleDepthBackend(MetricsMixin):
    """# MOCK: ground-truth Z-depth rendered from a ``SyntheticScene`` through the camera ``calib``.

    The frame pixels are ignored (only ``calib_version`` is checked); the scene is rendered with the P0 renderer.
    Optional seeded impairments (``numpy.random.default_rng(seed)``): additive Gaussian noise ``noise_sigma_m`` (m),
    constant ``bias_m`` (m) and multiplicative ``scale``, applied to the metric depth. ``kind='disparity'`` instead
    emits ``a / z + b`` (affine-invariant, like a relative model) with ``disp_a`` > 0 and ``disp_b``.
    Output host NumPy, caller-owned, CPU only (``device = "cpu"``).
    """

    def __init__(
        self,
        calib: CameraCalibration,
        scene: SyntheticScene | None = None,
        *,
        noise_sigma_m: float = 0.0,
        bias_m: float = 0.0,
        scale: float = 1.0,
        kind: Kind = "metric",
        disp_a: float = 1.0,
        disp_b: float = 0.0,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.calib = calib
        self.scene = scene or build_table_v1()
        self.noise_sigma_m, self.bias_m, self.scale = noise_sigma_m, bias_m, scale
        self.kind: Kind = kind
        self.disp_a, self.disp_b = disp_a, disp_b
        self._rng = np.random.default_rng(seed)
        self.device = "cpu"

    def warmup(self, shape: tuple[int, int]) -> None:
        """No-op (nothing to compile)."""

    def infer_raw(self, frame) -> RawDepth:
        """Rendered depth in the configured native space (``RawDepth``)."""
        if frame.calib_version != self.calib.calib_version:
            raise CalibVersionError(f"frame calib_version {frame.calib_version} != {self.calib.calib_version}")
        if frame.rgb.shape[:2] != (self.calib.height, self.calib.width):
            raise DepthError(f"frame {frame.rgb.shape[:2]} does not match calibration")

        def run() -> RawDepth:
            view = render_view(self.scene, self.calib)
            z = view.depth.astype(np.float64)
            valid = view.valid.copy()
            if self.noise_sigma_m > 0.0:
                z = z + self._rng.normal(0.0, self.noise_sigma_m, z.shape)
            z = self.scale * (z + self.bias_m)
            valid &= z > 0.0
            if self.kind == "metric":
                vals = np.where(valid, z, 0.0)
            else:
                vals = np.where(valid, self.disp_a / np.where(valid, z, 1.0) + self.disp_b, 0.0)
            return RawDepth(vals.astype(np.float32), valid, self.kind, frame.calib_version)

        return self._timed(run)

    def infer(self, frame) -> DepthResult:
        """Metric ``DepthResult`` (``kind='metric'`` only; conf 1.0 on valid pixels, a MOCK constant)."""
        raw = self.infer_raw(frame)
        res = require_infer_ready(raw)
        return DepthResult(res.z_depth_m, res.valid, np.where(res.valid, 1.0, 0.0).astype(np.float32), res.calib_version)
