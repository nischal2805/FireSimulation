"""Material table loaded from ``configs/materials.yaml`` (PRD 8.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from livefire.common.config import ConfigError, config_dir, load_yaml_text
from livefire.perception.material.types import MATERIAL_CLASSES

_FUEL = ("m0_areal", "T_ign", "t_delay", "mdot_ref", "dH", "chi", "char_yield", "y_soot", "consumed_when_burnt")
_COMMON = ("rho", "c", "k", "delta", "eps", "status")
_STATUSES = {"physical_estimate", "artistic"}



class MaterialTable:
    """Per-material properties indexed by ``material_id`` (0..7, order MATERIAL_CLASSES).

    Per-id float64 NumPy arrays (SI units, see materials.yaml header): rho, c, k, delta, eps,
    m0, m_res, T_ign, t_delay, mdot_ref, dH, chi, y_soot; bool arrays: flammable, consumed.
    Non-flammable ids have all fuel quantities 0 (m0 = m_res = 0). ``m_res = m0 * char_yield``
    (residual char mass, kg/m^2). Arrays are read-only."""

    def __init__(self, entries: dict[str, dict[str, Any]]) -> None:
        if list(entries) != list(MATERIAL_CLASSES):
            raise ConfigError(f"materials must be exactly {list(MATERIAL_CLASSES)} in order, got {list(entries)}")
        n = len(MATERIAL_CLASSES)
        z = lambda: np.zeros(n)
        self.names = MATERIAL_CLASSES
        self.rho, self.c, self.k, self.delta, self.eps = z(), z(), z(), z(), z()
        self.m0, self.m_res, self.T_ign, self.t_delay = z(), z(), z(), z()
        self.mdot_ref, self.dH, self.chi, self.y_soot = z(), z(), z(), z()
        self.flammable = np.zeros(n, dtype=bool)
        self.consumed = np.zeros(n, dtype=bool)
        self.status: list[str] = []
        for i, name in enumerate(MATERIAL_CLASSES):
            e = entries[name]
            if not isinstance(e, dict):
                raise ConfigError(f"{name}: entry must be a mapping")
            for f in _COMMON:
                if f not in e:
                    raise ConfigError(f"{name}: missing field '{f}'")
            if e["status"] not in _STATUSES:
                raise ConfigError(f"{name}: status must be one of {sorted(_STATUSES)}")
            if not isinstance(e.get("flammable"), bool):
                raise ConfigError(f"{name}: 'flammable' must be a bool")
            for f in ("rho", "c", "k", "delta", "eps"):
                self._num(name, f, e[f])
            if not 0 <= e["eps"] <= 1:
                raise ConfigError(f"{name}: eps must be in [0,1]")
            self.status.append(e["status"])
            self.rho[i], self.c[i], self.k[i] = e["rho"], e["c"], e["k"]
            self.delta[i], self.eps[i] = e["delta"], e["eps"]
            fuel_present = [f for f in _FUEL if f in e]
            if e["flammable"]:
                missing = [f for f in _FUEL if f not in e]
                if missing:
                    raise ConfigError(f"{name}: flammable entry missing fuel fields {missing}")
                for f in _FUEL[:-1]:
                    self._num(name, f, e[f], allow_zero=f == "char_yield")
                if not isinstance(e["consumed_when_burnt"], bool):
                    raise ConfigError(f"{name}: consumed_when_burnt must be a bool")
                for f in ("chi", "char_yield"):
                    if not 0 <= e[f] <= 1:
                        raise ConfigError(f"{name}: {f} must be in [0,1]")
                self.flammable[i] = True
                self.consumed[i] = e["consumed_when_burnt"]
                self.m0[i] = e["m0_areal"]
                self.m_res[i] = e["m0_areal"] * e["char_yield"]
                self.T_ign[i], self.t_delay[i], self.mdot_ref[i] = e["T_ign"], e["t_delay"], e["mdot_ref"]
                self.dH[i], self.chi[i], self.y_soot[i] = e["dH"], e["chi"], e["y_soot"]
            elif fuel_present:
                raise ConfigError(f"{name}: non-flammable entry must not have fuel fields {fuel_present}")
        self.C_areal = self.rho * self.c * self.delta  # J/(m^2 K)
        for a in vars(self).values():
            if isinstance(a, np.ndarray):
                a.flags.writeable = False

    @staticmethod
    def _num(name: str, field: str, v: Any, allow_zero: bool = False) -> None:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ConfigError(f"{name}.{field}: must be a finite number, got {v!r}")
        if v < 0 or (v == 0 and not allow_zero):
            raise ConfigError(f"{name}.{field}: must be positive, got {v}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> MaterialTable:
        """Load ``path`` (default ``<config dir>/materials.yaml``)."""
        p = Path(path) if path is not None else config_dir() / "materials.yaml"
        if not p.is_file():
            raise ConfigError(f"materials file not found: {p}")
        data = load_yaml_text(p.read_text())
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: root must be a mapping")
        return cls(data)

    def material_id(self, name: str) -> int:
        """Index of ``name`` in the fixed class order."""
        try:
            return MATERIAL_CLASSES.index(name)
        except ValueError:
            raise ConfigError(f"unknown material {name!r}") from None
