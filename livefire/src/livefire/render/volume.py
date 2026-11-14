"""Ray-march kernel (NVIDIA Warp) for the fire volume (PRD 7.4).

Per pixel of a (possibly reduced) view buffer, march a world-space ray through the cell-centred
fields T_g and soot (shape (G,G,G), index [ix,iy,iz], centre = roi_min + (i+0.5) h), sampled
trilinearly in world space. Output transmittance T_vol and added radiance L_vol (linear RGB).
Emission is absorbed by soot along the ray. Read-only on the fields. Device: any Warp device.
"""

from __future__ import annotations

import warp as wp

wp.config.quiet = True


@wp.func
def _corner(p: float, g: int):
    q = wp.clamp(p, 0.0, float(g - 1))
    i0 = wp.min(int(q), g - 2)
    return i0, q - float(i0)


@wp.func
def _trilinear(f: wp.array3d(dtype=wp.float32), p: wp.vec3, g: int):
    i, tx = _corner(p[0], g)
    j, ty = _corner(p[1], g)
    k, tz = _corner(p[2], g)
    c00 = f[i, j, k] * (1.0 - tx) + f[i + 1, j, k] * tx
    c10 = f[i, j + 1, k] * (1.0 - tx) + f[i + 1, j + 1, k] * tx
    c01 = f[i, j, k + 1] * (1.0 - tx) + f[i + 1, j, k + 1] * tx
    c11 = f[i, j + 1, k + 1] * (1.0 - tx) + f[i + 1, j + 1, k + 1] * tx
    return (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz


@wp.func
def _lut(lut: wp.array(dtype=wp.vec3), T: float, t_lo: float, t_hi: float):
    n = lut.shape[0]
    x = wp.clamp((T - t_lo) / (t_hi - t_lo), 0.0, 1.0) * float(n - 1)
    i0 = wp.min(int(x), n - 2)
    f = x - float(i0)
    return lut[i0] * (1.0 - f) + lut[i0 + 1] * f


@wp.kernel
def march_kernel(
    T_g: wp.array3d(dtype=wp.float32),
    soot: wp.array3d(dtype=wp.float32),
    lut: wp.array(dtype=wp.vec3),
    dirs: wp.array2d(dtype=wp.vec3),  # (Hs,Ws) unit world directions
    t0: wp.array2d(dtype=wp.float32),  # (Hs,Ws) march start, ray distance [m]
    t1: wp.array2d(dtype=wp.float32),  # (Hs,Ws) march end; t1 <= t0 marches nothing
    origin: wp.vec3,
    roi_min: wp.vec3,
    h: float,
    g: int,
    ds: float,
    t_min: float,
    emission_scale: float,
    T_emit_min: float,
    T_emit_ref: float,
    emission_power: float,
    k_abs: float,
    scatter: float,  # soot_albedo * ambient
    lut_lo: float,
    lut_hi: float,
    u_off: int,
    v_off: int,
    out_T: wp.array2d(dtype=wp.float32),
    out_L: wp.array2d(dtype=wp.vec3),
):
    iv, iu = wp.tid()  # launch dim = (rows, cols)
    u = iu + u_off
    v = iv + v_off
    ta = t0[v, u]
    tb = t1[v, u]
    tr = float(1.0)  # noqa: UP018 (Warp: mutable loop variable needs an explicit float)
    rad = wp.vec3(0.0, 0.0, 0.0)
    if tb > ta:
        d = dirs[v, u]
        t = ta
        while t < tb and tr >= t_min:
            seg = wp.min(ds, tb - t)
            pw = origin + d * (t + 0.5 * seg)
            gp = (pw - roi_min) / h - wp.vec3(0.5, 0.5, 0.5)
            Tk = _trilinear(T_g, gp, g)
            s = _trilinear(soot, gp, g)
            sig = k_abs * wp.max(s, 0.0)
            f = wp.max((Tk - T_emit_min) / (T_emit_ref - T_emit_min), 0.0)
            if f > 0.0:
                le = emission_scale * wp.pow(f, emission_power)
                rad = rad + _lut(lut, Tk, lut_lo, lut_hi) * (le * seg * tr)
            if sig > 0.0:
                rad = rad + wp.vec3(1.0, 1.0, 1.0) * (scatter * sig * seg * tr)
                tr = tr * wp.exp(-sig * seg)
            t = t + seg
    out_T[v, u] = tr
    out_L[v, u] = rad
