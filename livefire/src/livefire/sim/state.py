"""Host (NumPy) copy of the dynamic fire state (PRD 6.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class FireStateHost:
    """Host copy of FireState. Per element (N_e,): T_s [K], ignition_exposure [s], m_fuel
    [kg/m^2], char [0..1], removed [0..1], burning u8. Grid (G,G,G): T_g [K], soot, p [Pa], flame [W/m^3] (combustion intensity, ADR-0004);
    ``vel`` (G,G,G,3) m/s cell-centred, world axes. Float arrays are float32. The caller owns
    the copy (it does not alias device memory)."""

    T_s: np.ndarray
    ignition_exposure: np.ndarray
    m_fuel: np.ndarray
    char: np.ndarray
    removed: np.ndarray
    burning: np.ndarray
    T_g: np.ndarray
    vel: np.ndarray
    soot: np.ndarray
    p: np.ndarray
    flame: np.ndarray
    t_sim: float
    step: int

    _ARRAYS = (
        "T_s", "ignition_exposure", "m_fuel", "char", "removed", "burning",
        "T_g", "vel", "soot", "p", "flame",
    )  # fmt: skip

    def digest(self, decimals: int = 6) -> str:
        """sha256 over arrays rounded to ``decimals`` (names, shapes, then bytes), plus step and
        rounded t_sim. Used for replay determinism checks."""
        h = hashlib.sha256()
        for name in self._ARRAYS:
            a = np.ascontiguousarray(np.round(getattr(self, name).astype(np.float64), decimals))
            a = a + 0.0  # normalise -0.0 to 0.0
            h.update(f"{name}{a.shape}".encode())
            h.update(a.tobytes())
        h.update(f"step={int(self.step)};t={round(float(self.t_sim), decimals)}".encode())
        return h.hexdigest()
