import dataclasses

import numpy as np
import pytest

from livefire.capture.types import Frame
from livefire.common import interfaces
from livefire.common.geometry import AABB, CameraCalibration, look_at, make_K
from livefire.perception.material.types import MATERIAL_CLASSES, MaterialPrediction
from livefire.perception.types import DepthResult
from livefire.scene.snapshot import (
    CalibVersionError,
    ObjectRegistry,
    SceneSnapshot,
    WorldObject,
    assert_calib_version,
)  # fmt: skip
from livefire.sim.state import FireStateHost


def _pred(p=None, source="model"):
    p = np.eye(8, dtype=np.float32)[2] if p is None else p
    return MaterialPrediction(p, "table", 0.9, source, False)


def _cam(v=1):
    return CameraCalibration(0, 8, 6, make_K(10, 10, 4, 3), np.zeros(5), look_at((1, 0, 1), (0, 0, 0)), v)


def _snap(cam_version=1, n=4, g=4):
    return SceneSnapshot(
        version=1, calib_version=1, roi=AABB(np.array([-1.0, -1, -0.1]), np.array([1.0, 1, 1.9])),
        h_gas=0.5, cameras={0: _cam(cam_version)},
        pos=np.zeros((n, 3), np.float32), normal=np.tile(np.float32([0, 0, 1]), (n, 1)),
        area=np.full(n, 1e-4, np.float32), obj_id=np.zeros(n, np.int32),
        material_id=np.zeros(n, np.uint8), material_conf=np.ones(n, np.float32),
        neighbors=np.full((n, 3), -1, np.int32), solid_mask=np.zeros((g, g, g), np.uint8),
        solid_fraction=np.zeros((g, g, g), np.float32),
        view_depth={0: np.zeros((6, 8), np.float32)}, view_hole_mask={0: np.ones((6, 8), bool)},
        objects=ObjectRegistry([WorldObject(0, "a", _pred())]), surfel_spacing=0.01,
    )  # fmt: skip


def _depth(**kw):
    z = np.ones((3, 4), np.float32)
    args = {"z_depth_m": z, "valid": np.ones((3, 4), bool), "conf": np.ones((3, 4), np.float32), "calib_version": 1}
    args.update(kw)
    return DepthResult(**args)


def test_frame_validation():
    Frame(0, np.zeros((2, 3, 3), np.uint8), 1, 0, 1)
    with pytest.raises(ValueError):
        Frame(0, np.zeros((2, 3, 3), np.float32), 1, 0, 1)
    with pytest.raises(ValueError):
        Frame(0, np.zeros((2, 3), np.uint8), 1, 0, 1)


def test_depth_result_validation():
    _depth()
    with pytest.raises(ValueError):
        _depth(z_depth_m=np.ones((3, 4), np.float64))
    with pytest.raises(ValueError):
        _depth(valid=np.ones((3, 3), bool))
    z = np.ones((3, 4), np.float32)
    z[0, 0] = np.nan
    with pytest.raises(ValueError):
        _depth(z_depth_m=z)
    valid = np.ones((3, 4), bool)
    valid[1, 1] = False  # z there is 1.0, must be 0.0
    with pytest.raises(ValueError):
        _depth(valid=valid)
    z2 = np.ones((3, 4), np.float32)
    z2[1, 1] = 0.0
    _depth(z_depth_m=z2, valid=valid)


def test_material_prediction_validation():
    _pred()
    with pytest.raises(ValueError):
        _pred(np.full(8, 0.2, np.float32))
    with pytest.raises(ValueError):
        _pred(np.eye(8)[0])  # float64
    with pytest.raises(ValueError):
        _pred(source="bogus")
    p = np.eye(8, dtype=np.float32)[0] * (1 + 5e-5)
    _pred(p)  # within tolerance


def test_snapshot_valid_and_readonly():
    s = _snap()
    for a in (s.pos, s.neighbors, s.solid_mask, s.view_depth[0], s.view_hole_mask[0]):
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.flat[0] = a.flat[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.version = 2  # type: ignore[misc]


def test_snapshot_bad_dtype_shape():
    s = _snap()
    kw = {f.name: getattr(s, f.name) for f in dataclasses.fields(s)}
    with pytest.raises(ValueError, match="pos"):
        SceneSnapshot(**{**kw, "pos": np.zeros((4, 3), np.float64)})
    with pytest.raises(ValueError, match="area"):
        SceneSnapshot(**{**kw, "area": np.zeros(5, np.float32)})
    with pytest.raises(ValueError, match="solid_mask"):
        SceneSnapshot(**{**kw, "solid_mask": np.zeros((4, 4, 3), np.uint8)})
    with pytest.raises(ValueError, match="neighbors"):
        SceneSnapshot(**{**kw, "neighbors": np.full((4, 3), 9, np.int32)})


def test_calib_mismatch_raises():
    with pytest.raises(CalibVersionError):
        _snap(cam_version=2)
    s = _snap()
    assert_calib_version(s, Frame(0, np.zeros((6, 8, 3), np.uint8), 0, 0, 1))
    with pytest.raises(CalibVersionError):
        assert_calib_version(s, Frame(0, np.zeros((6, 8, 3), np.uint8), 0, 0, 2))


def test_override_material():
    reg = ObjectRegistry([WorldObject(7, "x", _pred())])
    reg.override_material(7, "paper")
    m = reg[7].material
    assert m.source == "manual_override" and m.material_id == MATERIAL_CLASSES.index("paper")
    with pytest.raises(ValueError):
        reg.override_material(7, "unobtainium")


def _state(n=5, g=3):
    r = np.random.default_rng(0)
    f = lambda *s: r.random(s).astype(np.float32)
    return FireStateHost(f(n), f(n), f(n), f(n), f(n), np.ones(n, np.uint8), f(g, g, g),
                         f(g, g, g, 3), f(g, g, g), f(g, g, g), 0.5, 3)  # fmt: skip


def test_digest_stable_and_sensitive():
    a, b = _state(), _state()
    assert a.digest() == b.digest()
    b.T_g[0, 0, 0] += 1e-3
    assert a.digest() != b.digest()
    c = _state()
    c.step += 1
    assert a.digest() != c.digest()


def test_protocols_have_documented_checklist():
    for name in ("CameraSource", "DepthBackend", "SegmentationBackend", "MaterialBackend",
                 "SceneReconstructor", "SimulationBackend", "RendererBackend",
                 "ReplayController", "BenchmarkRecorder", "FireStateView"):  # fmt: skip
        doc = getattr(interfaces, name).__doc__
        assert doc and "calib_version" in doc, name
    for name in ("DepthBackend", "SegmentationBackend", "MaterialBackend", "SceneReconstructor",
                 "SimulationBackend", "RendererBackend"):  # fmt: skip
        cls = getattr(interfaces, name)
        for prop in ("latency_ms", "peak_vram_allocated_mb", "peak_vram_reserved_mb"):
            assert hasattr(cls, prop), (name, prop)
