import pytest
import yaml

from livefire.common.config import ConfigError, config_dir
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.sim.materials import MaterialTable


@pytest.fixture
def raw():
    return yaml.safe_load((config_dir() / "materials.yaml").read_text())


def _dump(tmp_path, d):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def test_ids_and_derived():
    t = MaterialTable.load(config_dir() / "materials.yaml")
    assert MATERIAL_CLASSES == (
        "paper",
        "cloth",
        "wood",
        "plastic",
        "metal",
        "glass",
        "ceramic",
        "other_nonflammable",
    )
    assert [t.material_id(n) for n in MATERIAL_CLASSES] == list(range(8))
    assert t.flammable.tolist() == [True] * 4 + [False] * 4
    assert t.m_res[0] == 0.0
    assert t.m_res[2] == pytest.approx(0.15 * 3.0)
    assert (t.m0[4:] == 0).all() and (t.m_res[4:] == 0).all() and (t.T_ign[4:] == 0).all()
    assert t.C_areal[2] == pytest.approx(600 * 1700 * 5e-3)
    assert t.consumed.tolist()[:4] == [True, True, False, False]
    assert not t.rho.flags.writeable


def test_default_path_loads():
    assert MaterialTable.load().rho.shape == (8,)


def test_nonflammable_with_fuel_field_rejected(tmp_path, raw):
    raw["metal"]["m0_areal"] = 1.0
    with pytest.raises(ConfigError, match="metal"):
        MaterialTable.load(_dump(tmp_path, raw))


def test_flammable_missing_fuel_field_rejected(tmp_path, raw):
    del raw["wood"]["dH"]
    with pytest.raises(ConfigError, match="dH"):
        MaterialTable.load(_dump(tmp_path, raw))


@pytest.mark.parametrize("mat,field,val", [
    ("paper", "chi", 1.5), ("paper", "char_yield", -0.1), ("glass", "eps", 1.2),
    ("wood", "rho", -5), ("wood", "T_ign", 0), ("metal", "k", 0),
])  # fmt: skip
def test_invalid_values_rejected(tmp_path, raw, mat, field, val):
    raw[mat][field] = val
    with pytest.raises(ConfigError, match=mat):
        MaterialTable.load(_dump(tmp_path, raw))


def test_missing_common_field_and_bad_order(tmp_path, raw):
    r2 = {k: dict(v) for k, v in raw.items()}
    del r2["glass"]["status"]
    with pytest.raises(ConfigError, match="status"):
        MaterialTable.load(_dump(tmp_path, r2))
    r3 = dict(reversed(list(raw.items())))
    with pytest.raises(ConfigError):
        MaterialTable.load(_dump(tmp_path, r3))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        MaterialTable.load(tmp_path / "nope.yaml")


def test_status_detail_present_for_every_material_and_valid():
    t = MaterialTable.load()
    for name, d in zip(MATERIAL_CLASSES, t.status_detail, strict=True):
        assert d, f"{name}: status_detail missing"
        assert set(d) <= {"thermal", "ignition", "combustion", "radiation"}
        assert set(d.values()) <= {"physical_estimate", "artistic"}
        assert {"thermal", "radiation"} <= set(d)


def test_status_detail_absent_is_accepted(tmp_path, raw):
    for e in raw.values():
        e.pop("status_detail", None)
    t = MaterialTable.load(_dump(tmp_path, raw))
    assert t.status_detail == [{}] * 8


@pytest.mark.parametrize("detail,match", [
    ({"thermal": "artistic", "bogus": "artistic"}, "bogus"),
    ({"thermal": "measured"}, "thermal"),
    ("artistic", "status_detail"),
])  # fmt: skip
def test_status_detail_invalid_rejected(tmp_path, raw, detail, match):
    raw["wood"]["status_detail"] = detail
    with pytest.raises(ConfigError, match=match):
        MaterialTable.load(_dump(tmp_path, raw))
