"""P1-004: table-plane anchoring recovers a known scale (and shift) from a synthetic plane with outliers."""

import numpy as np
import pytest

from livefire.capture.types import Frame
from livefire.common.errors import CalibVersionError, DepthError
from livefire.perception.depth import (
    AnchorConfig,
    AnchoredDepthBackend,
    OracleDepthBackend,
    TableRegion,
    fit_anchor,
    table_plane_depth,
)
from livefire.scene.synthetic import load_cameras

CAM = load_cameras()[0]
REGION = TableRegion.from_footprint((0.6, 0.4), 0.0)
PLANE_Z, MASK = table_plane_depth(CAM, REGION)


def test_plane_depth_matches_geometry():
    assert MASK.sum() > 5000
    # back-project the plane depth: every masked pixel must land on z = 0 inside the polygon
    from livefire.common.geometry import backproject

    v, u = np.nonzero(MASK)
    p = backproject(CAM, np.stack([u, v], 1).astype(float), PLANE_Z[v, u])
    assert np.abs(p[:, 2]).max() < 1e-9
    assert np.abs(p[:, 0]).max() <= 0.6 + 1e-9 and np.abs(p[:, 1]).max() <= 0.4 + 1e-9


def _outliers(raw, frac, rng, lo, hi):
    out = raw.copy()
    idx = np.flatnonzero(MASK.ravel())
    pick = rng.choice(idx, int(frac * idx.size), replace=False)
    out.ravel()[pick] = rng.uniform(lo, hi, pick.size)
    return out


def test_metric_scale_recovered_within_1pct_with_outliers():
    rng = np.random.default_rng(0)
    s_true = 1.37
    raw = np.where(MASK, PLANE_Z / s_true, 3.0).astype(np.float32)  # z = s * raw
    raw = _outliers(raw, 0.25, rng, 0.05, 2.0).astype(np.float32)
    fit = fit_anchor(raw, np.ones_like(MASK), PLANE_Z, MASK, "metric", AnchorConfig(min_inlier_ratio=0.5))
    assert fit.scale == pytest.approx(s_true, rel=0.01)
    assert 0.7 < fit.inlier_ratio < 0.8
    assert fit.resid_mae_m < 0.01


def test_disparity_scale_shift_recovered_within_1pct_with_outliers():
    rng = np.random.default_rng(1)
    a_true, b_true = 2.5, 0.4
    raw = np.where(MASK, a_true / np.where(MASK, PLANE_Z, 1.0) + b_true, 0.0).astype(np.float32)
    raw = _outliers(raw, 0.25, rng, 0.0, 6.0).astype(np.float32)
    fit = fit_anchor(raw, np.ones_like(MASK), PLANE_Z, MASK, "disparity", AnchorConfig())
    assert fit.scale == pytest.approx(a_true, rel=0.01)
    assert fit.shift == pytest.approx(b_true, rel=0.01)
    assert fit.resid_mae_m < 0.02


def test_rejects_low_inlier_ratio():
    rng = np.random.default_rng(2)
    raw = np.where(MASK, rng.uniform(0.1, 5.0, MASK.shape), 1.0).astype(np.float32)  # pure noise on the table
    with pytest.raises(DepthError, match="inlier"):
        fit_anchor(raw, np.ones_like(MASK), PLANE_Z, MASK, "metric", AnchorConfig(min_inlier_ratio=0.5))


def test_rejects_too_few_pixels():
    tiny = np.zeros_like(MASK)
    tiny[:5, :5] = True
    with pytest.raises(DepthError, match="table pixels"):
        fit_anchor(np.ones(MASK.shape, np.float32), np.ones_like(MASK), PLANE_Z, tiny, "metric")


def test_inlier_ratio_threshold_is_configurable():
    rng = np.random.default_rng(3)
    raw = np.where(MASK, PLANE_Z / 1.2, 1.0).astype(np.float32)
    raw = _outliers(raw, 0.4, rng, 0.05, 2.0).astype(np.float32)
    ok = np.ones_like(MASK)
    fit_anchor(raw, ok, PLANE_Z, MASK, "metric", AnchorConfig(min_inlier_ratio=0.5))
    with pytest.raises(DepthError):
        fit_anchor(raw, ok, PLANE_Z, MASK, "metric", AnchorConfig(min_inlier_ratio=0.7))


def _frame(calib_version=1):
    return Frame(0, np.zeros((CAM.height, CAM.width, 3), np.uint8), 0, 0, calib_version)


@pytest.mark.parametrize(
    "kw", [{"kind": "metric", "scale": 0.6}, {"kind": "disparity", "disp_a": 3.0, "disp_b": 0.7}]
)
def test_anchored_oracle_recovers_ground_truth(kw):
    """End to end: a mis-scaled / affine oracle anchored on the table gives back the true metric depth."""
    from livefire.scene.synthetic import build_table_v1, render_view

    truth = render_view(build_table_v1(), CAM)
    be = AnchoredDepthBackend(OracleDepthBackend(CAM, **kw), CAM, REGION, AnchorConfig())
    res = be.infer(_frame())
    both = truth.valid & res.valid
    rel = np.abs(res.z_depth_m[both] - truth.depth[both]) / truth.depth[both]
    assert np.median(rel) < 0.005
    assert be.last_fit.inlier_ratio > 0.8
    assert res.calib_version == 1 and np.isfinite(res.z_depth_m).all()


def test_anchored_backend_calib_version_mismatch():
    be = AnchoredDepthBackend(OracleDepthBackend(CAM), CAM, REGION, AnchorConfig())
    with pytest.raises(CalibVersionError):
        be.infer(_frame(calib_version=99))


# ---- anchor guards (review batch D1/D2/D3)


def test_non_positive_scale_rejected():
    # disparity that grows with depth => negative scale a
    d = np.where(MASK, -2.0 / np.where(MASK, PLANE_Z, 1.0) + 5.0, 0.0).astype(np.float32)
    with pytest.raises(DepthError, match="non-positive scale"):
        fit_anchor(d, np.ones_like(MASK), PLANE_Z, MASK, "disparity", AnchorConfig())


def test_ill_conditioned_table_span_rejected_for_disparity():
    # plane_z nearly constant over the table: scale and shift are not separable
    zp = np.where(MASK, 1.5 + 1e-4 * np.arange(MASK.size).reshape(MASK.shape) / MASK.size, 0.0)
    raw = np.where(MASK, 2.0 / np.where(MASK, zp, 1.0) + 0.3, 0.0).astype(np.float32)
    with pytest.raises(DepthError, match="too little depth range"):
        fit_anchor(raw, np.ones_like(MASK), zp, MASK, "disparity", AnchorConfig())


def test_apply_anchor_masks_raw_below_shift():
    from livefire.perception.depth import RawDepth, apply_anchor
    from livefire.perception.depth.anchor import AnchorFit

    fit = AnchorFit("disparity", 2.0, 0.5, 1.0, 10, 10, 0.0, 0.0, 0.0, 0.0)
    vals = np.array([[0.2, 0.5, 0.6, 1.5]], np.float32)  # raw <= shift is invalid; raw - shift tiny is beyond z_max
    z, valid = apply_anchor(RawDepth(vals, np.ones_like(vals, bool), "disparity", 1), fit, z_max_m=20.0)
    assert valid.tolist() == [[False, False, True, True]]
    assert z[0, 0] == 0.0 and z[0, 1] == 0.0 and z.dtype == np.float32
    assert z[0, 2] == pytest.approx(2.0 / 0.1) and z[0, 3] == pytest.approx(2.0)


def test_same_seed_ransac_is_repeatable_and_seed_matters_only_in_sampling():
    rng = np.random.default_rng(4)
    raw = _outliers(np.where(MASK, PLANE_Z / 1.3, 1.0).astype(np.float32), 0.3, rng, 0.05, 2.0).astype(np.float32)
    ones = np.ones_like(MASK)
    cfg = AnchorConfig(max_fit_points=3000)  # forces the seeded subsample path
    f1 = fit_anchor(raw, ones, PLANE_Z, MASK, "metric", cfg)
    f2 = fit_anchor(raw, ones, PLANE_Z, MASK, "metric", cfg)
    assert f1 == f2
    f3 = fit_anchor(raw, ones, PLANE_Z, MASK, "metric", AnchorConfig(max_fit_points=3000, seed=99))
    assert f3.scale == pytest.approx(f1.scale, rel=0.01)


def test_residual_stats_are_finite_when_disparity_denominator_is_non_positive():
    # a few table pixels with raw below the fitted shift give z_est = inf; stats must still be finite
    a, b = 2.5, 0.4
    raw = np.where(MASK, a / np.where(MASK, PLANE_Z, 1.0) + b, 0.0).astype(np.float32)
    idx = np.flatnonzero(MASK.ravel())[:200]
    raw.ravel()[idx] = 0.1  # < shift
    fit = fit_anchor(raw, np.ones_like(MASK), PLANE_Z, MASK, "disparity", AnchorConfig())
    for v in (fit.resid_mae_m, fit.resid_rms_m, fit.resid_p95_m, fit.resid_mae_all_m):
        assert np.isfinite(v)
    assert fit.resid_mae_all_m > fit.resid_mae_m  # outliers count against the all-pixels metric only


def test_yaml_anchor_config_exposes_all_fields():
    from livefire.perception.depth.base import load_depth_config
    cfg = AnchorConfig.from_yaml(load_depth_config())
    assert cfg.min_disparity_span_rel == 0.05 and cfg.max_fit_points == 20000 and cfg.z_max_m == 20.0
    assert TableRegion.from_config().polygon_xy.shape == (4, 2)


def test_import_without_torch_or_transformers():
    """`import livefire` and `livefire.perception.depth` must work when torch/transformers cannot be imported."""
    import subprocess
    import sys
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'transformers'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import livefire, livefire.perception.depth\n"
        "from livefire.perception.depth.hf_backends import depth_extra_available, create_depth_backend\n"
        "assert not depth_extra_available()\n"
        "assert not any(m.split('.')[0] in ('torch', 'transformers') for m in sys.modules)\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
