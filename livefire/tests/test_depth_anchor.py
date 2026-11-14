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
