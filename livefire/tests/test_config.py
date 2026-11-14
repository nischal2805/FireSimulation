import pytest

from livefire.common.config import ConfigError, Params, load_params, load_yaml


@pytest.mark.parametrize("name", ["app", "grid", "camera", "sim", "render", "materials"])
def test_loads_all(name):
    assert isinstance(load_yaml(name), dict)


def test_values():
    assert load_yaml("app")["seed"] == 1234
    assert load_yaml("grid")["G"] == 96
    assert load_yaml("materials")["paper"]["T_ign"] == 505
    assert load_params("sim").keys() == []


def test_env_dir(tmp_path, monkeypatch):
    (tmp_path / "x.yaml").write_text("h_c: {value: 10.0, unit: 'W/m^2/K', range: [1, 50], status: artistic}\n")
    monkeypatch.setenv("LIVEFIRE_CONFIG_DIR", str(tmp_path))
    p = load_params("x")
    assert p["h_c"] == 10.0
    assert p.meta("h_c")["unit"] == "W/m^2/K"


GOOD = {"value": 1.0, "unit": "m", "range": [0, 2], "status": "artistic"}


def test_good():
    assert Params({"a": GOOD})["a"] == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "status": "guess"},
        {k: v for k, v in GOOD.items() if k != "unit"},
        {**GOOD, "value": 3.0},
    ],
)
def test_bad_raises_with_key(bad):
    with pytest.raises(ConfigError, match="a"):
        Params({"a": bad})
