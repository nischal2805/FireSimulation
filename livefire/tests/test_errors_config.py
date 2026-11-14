import pytest

from livefire.common import errors
from livefire.common.config import config_dir, load_yaml_text
from livefire.scene.snapshot import CalibVersionError


def test_error_types_importable():
    assert CalibVersionError is errors.CalibVersionError
    for n in ("DepthError", "SegmentationError", "MaterialError", "ReconstructionError"):
        assert issubclass(getattr(errors, n), Exception)


def test_yaml12_float_loads_as_float():
    d = load_yaml_text("{x: 1.4e7, y: 1.0e-4, z: 800, w: 2.5}")
    assert d["x"] == pytest.approx(1.4e7) and isinstance(d["x"], float)
    assert isinstance(d["y"], float) and isinstance(d["z"], int)


def test_load_yaml_uses_it(tmp_path, monkeypatch):
    (tmp_path / "t.yaml").write_text("{x: 1.4e7}")
    monkeypatch.setenv("LIVEFIRE_CONFIG_DIR", str(tmp_path))
    from livefire.common.config import load_yaml

    assert load_yaml("t") == {"x": 1.4e7}
    assert config_dir() == tmp_path
