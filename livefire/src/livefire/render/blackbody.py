"""Blackbody colour LUT: Planck spectrum integrated against CIE 1931 2-degree CMFs -> linear sRGB.

The CMFs use the multi-lobe Gaussian analytic fit of Wyman, Sloan and Shirley (JCGT 2013) to the
CIE 1931 2-degree observer (not the tabulated data). Each entry is normalised to unit luminance
(Y = 1, so only chromaticity is stored; intensity comes from the emission law). Host NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_LUT = 64
T_LUT_MIN = 800.0  # K
T_LUT_MAX = 3000.0  # K

_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
_XYZ_TO_SRGB = np.array(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]
)


def _g(lam: NDArray, mu: float, s1: float, s2: float) -> NDArray:
    t = (lam - mu) / np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * t * t)


def cie_xyz_bar(lam_nm: NDArray) -> NDArray:
    """CIE 1931 2-degree CMFs (Wyman 2013 fit) at wavelengths (M,) nm -> (M,3)."""
    x = 1.056 * _g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=1)


def blackbody_rgb(T: float) -> NDArray[np.float64]:
    """Linear sRGB (3,) of a blackbody at T kelvin, negatives clipped, normalised to luminance 1."""
    lam = np.arange(360.0, 831.0, 1.0)
    lm = lam * 1e-9
    planck = 2 * _H * _C**2 / lm**5 / np.expm1(_H * _C / (lm * _KB * T))
    xyz = planck @ cie_xyz_bar(lam)
    rgb = np.clip(_XYZ_TO_SRGB @ xyz, 0.0, None)
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return rgb / y


def blackbody_lut(n: int = N_LUT, t_min: float = T_LUT_MIN, t_max: float = T_LUT_MAX) -> NDArray[np.float32]:
    """LUT (n,3) float32 of unit-luminance linear sRGB, uniform in T from t_min to t_max K."""
    return np.stack([blackbody_rgb(t) for t in np.linspace(t_min, t_max, n)]).astype(np.float32)
