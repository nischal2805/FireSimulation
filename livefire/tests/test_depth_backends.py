"""P1-004: depth backend contract. HF backends run only when the `depth` extra and cached weights exist."""

import numpy as np
import pytest

from livefire.capture.types import Frame
from livefire.common.errors import DepthError
from livefire.common.interfaces import DepthBackend
from livefire.perception.depth import OracleDepthBackend
from livefire.perception.depth.base import model_cache_dir
from livefire.perception.depth.hf_backends import HFDepthBackend, depth_extra_available
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

CAM = load_cameras()[0]


def _frame():
    v = render_view(build_table_v1(), CAM)
    return Frame(0, v.rgb, 0, 0, CAM.calib_version), v


def test_oracle_contract():
    fr, v = _frame()
    be = OracleDepthBackend(CAM)
    assert isinstance(be, DepthBackend)
    be.warmup((CAM.height, CAM.width))
    r = be.infer(fr)
    assert r.z_depth_m.shape == (CAM.height, CAM.width) and r.z_depth_m.dtype == np.float32
    assert r.z_depth_m.flags.c_contiguous and np.isfinite(r.z_depth_m).all()
    assert (r.z_depth_m[~r.valid] == 0.0).all()
    np.testing.assert_array_equal(r.z_depth_m, v.depth)
    assert r.calib_version == CAM.calib_version
    assert be.latency_ms > 0 and be.peak_vram_allocated_mb == 0.0 and be.peak_vram_reserved_mb == 0.0


def test_oracle_noise_bias_seeded_and_invalid_stays_zero():
    fr, v = _frame()
    a = OracleDepthBackend(CAM, noise_sigma_m=0.01, bias_m=0.02, seed=7).infer(fr)
    b = OracleDepthBackend(CAM, noise_sigma_m=0.01, bias_m=0.02, seed=7).infer(fr)
    np.testing.assert_array_equal(a.z_depth_m, b.z_depth_m)
    ok = v.valid & a.valid
    assert abs(float((a.z_depth_m[ok] - v.depth[ok]).mean()) - 0.02) < 1e-3
    assert (a.z_depth_m[~a.valid] == 0.0).all() and np.isfinite(a.z_depth_m).all()


def test_oracle_disparity_infer_requires_anchor():
    fr, _ = _frame()
    with pytest.raises(DepthError, match="anchor"):
        OracleDepthBackend(CAM, kind="disparity").infer(fr)


_HF = [
    ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "metric"),
    ("depth-anything/Depth-Anything-V2-Small-hf", "disparity"),
]


@pytest.mark.parametrize(("hf_id", "kind"), _HF)
def test_hf_backend_output_contract(hf_id, kind):
    if not depth_extra_available():
        pytest.skip("depth extra (torch, transformers) not installed")
    cached = list(model_cache_dir().glob(f"models--{hf_id.replace('/', '--')}"))
    if not cached:
        pytest.skip(f"weights for {hf_id} not cached under data/models (run `make depth-spike`)")
    be = HFDepthBackend(hf_id, kind, device="cpu", input_size=266)
    img = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    be.warmup((120, 160))
    raw = be.infer_raw(Frame(0, img, 0, 0, 3))
    assert raw.values.shape == (120, 160) and raw.values.dtype == np.float32
    assert np.isfinite(raw.values).all() and raw.calib_version == 3 and raw.kind == kind
    assert be.latency_ms > 0 and be.peak_vram_allocated_mb == 0.0
    if kind == "metric":
        r = be.infer(Frame(0, img, 0, 0, 3))
        assert (r.z_depth_m[~r.valid] == 0.0).all() and r.valid.any()
    else:
        with pytest.raises(DepthError):
            be.infer(Frame(0, img, 0, 0, 3))
