"""YAML config loading and validated physical/artistic parameter access."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_STATUSES = {"physical_estimate", "artistic"}
_FIELDS = ("value", "unit", "range", "status")


class Yaml12Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML 1.2 floats such as ``1.4e7`` (PyYAML needs ``1.4e+7``)."""


Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?$"),
    list("-+0123456789."),
)


def load_yaml_text(text: str) -> Any:
    """Parse YAML text with ``Yaml12Loader`` (safe; YAML 1.2 float literals)."""
    return yaml.load(text, Loader=Yaml12Loader)


class ConfigError(ValueError):
    """Raised for missing files or invalid config entries; message names the key."""


def config_dir() -> Path:
    """Config directory: env LIVEFIRE_CONFIG_DIR, else `<project root>/configs`."""
    env = os.environ.get("LIVEFIRE_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    """Load `configs/<name>.yaml` as a dict (empty file or `{}` gives `{}`)."""
    path = config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = load_yaml_text(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: root must be a mapping")
    return data


class Params:
    """Validated `{value, unit, range, status}` entries. Values are Python floats, units as declared."""

    def __init__(self, entries: dict[str, Any]) -> None:
        for key, e in entries.items():
            _validate(key, e)
        self._entries = entries

    def __getitem__(self, key: str) -> float:
        return float(self.meta(key)["value"])

    def meta(self, key: str) -> dict[str, Any]:
        """Full entry dict (value, unit, range, status) for `key`."""
        try:
            return self._entries[key]
        except KeyError:
            raise ConfigError(f"unknown parameter: {key}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def keys(self) -> list[str]:
        return list(self._entries)


def _validate(key: str, e: Any) -> None:
    if not isinstance(e, dict):
        raise ConfigError(f"{key}: entry must be a mapping with {_FIELDS}")
    for f in _FIELDS:
        if f not in e:
            raise ConfigError(f"{key}: missing field '{f}'")
    if e["status"] not in _STATUSES:
        raise ConfigError(f"{key}: status must be one of {sorted(_STATUSES)}, got {e['status']!r}")
    rng = e["range"]
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        raise ConfigError(f"{key}: range must be [lo, hi]")
    v = e["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{key}: value must be a number, got {v!r}")
    if not (rng[0] <= v <= rng[1]):
        raise ConfigError(f"{key}: value {v} outside range {list(rng)}")


def load_params(name: str) -> Params:
    """Load `configs/<name>.yaml` and validate every entry as a parameter."""
    return Params(load_yaml(name))
