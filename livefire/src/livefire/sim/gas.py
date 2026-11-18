"""Gas solver baseline (PRD 8.4): MAC grid, semi-Lagrangian advection, Boussinesq buoyancy, vorticity
confinement, red-black Gauss-Seidel pressure projection. NVIDIA Warp kernels (ADR-0001), float32.

Staggering (ADR-0002). Grid is ``G^3`` cells of size ``h`` [m] over the ROI. In *cell units* with the origin
at ``roi_min`` (index space), cell ``(i,j,k)`` has its centre at ``(i+.5, j+.5, k+.5)``; the faces are
``u (G+1,G,G)`` at ``(i, j+.5, k+.5)``, ``v (G,G+1,G)`` at ``(i+.5, j, k+.5)`` and ``w (G,G,G+1)`` at
``(i+.5, j+.5, k)``. ``T_g``, ``soot``, ``p`` are cell-centred ``(G,G,G)``. All units SI; world axes.

Boundaries: domain walls and faces touching a solid cell are no-through (fixed face velocity 0); the -x wall
faces carry the (default zero) wind inflow. The top face ``z = roi_max`` is open: pressure Dirichlet
``p = 0`` in a ghost cell, outflow allowed. Pressure is stored kinematically (``p / rho_g``, m^2/s^2).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from livefire.common.config import Params
from livefire.sim.device import select_device, to_device

_F32 = wp.float32


# --------------------------------------------------------------------------------------------------
# interpolation helpers (index space, clamp-to-domain)
# --------------------------------------------------------------------------------------------------
@wp.func
def _cl(q: float, n: int):
    qc = wp.clamp(q, 0.0, float(n - 1))
    i0 = wp.min(int(qc), n - 2)
    return i0, qc - float(i0)


@wp.func
def _samp(f: wp.array3d(dtype=wp.float32), p: wp.vec3, o: wp.vec3):
    """Trilinear sample of ``f`` (value ``f[0,0,0]`` located at index-space point ``o``) at point ``p``."""
    i, tx = _cl(p[0] - o[0], f.shape[0])
    j, ty = _cl(p[1] - o[1], f.shape[1])
    k, tz = _cl(p[2] - o[2], f.shape[2])
    c00 = f[i, j, k] * (1.0 - tx) + f[i + 1, j, k] * tx
    c10 = f[i, j + 1, k] * (1.0 - tx) + f[i + 1, j + 1, k] * tx
    c01 = f[i, j, k + 1] * (1.0 - tx) + f[i + 1, j, k + 1] * tx
    c11 = f[i, j + 1, k + 1] * (1.0 - tx) + f[i + 1, j + 1, k + 1] * tx
    return (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz


@wp.func
def _vel(u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32), p: wp.vec3):
    return wp.vec3(
        _samp(u, p, wp.vec3(0.0, 0.5, 0.5)),
        _samp(v, p, wp.vec3(0.5, 0.0, 0.5)),
        _samp(w, p, wp.vec3(0.5, 0.5, 0.0)),
    )


@wp.func
def _backtrace(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    x: wp.vec3,
    c: float,
):
    """RK2 (midpoint) backtrace; ``c = dt / h`` converts velocity to cells per step."""
    v1 = _vel(u, v, w, x)
    xm = x - 0.5 * c * v1
    v2 = _vel(u, v, w, xm)
    return x - c * v2


# --------------------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------------------
@wp.kernel
def advect_u_k(
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32), c: float,
):  # fmt: skip
    i, j, k = wp.tid()
    x = wp.vec3(float(i), float(j) + 0.5, float(k) + 0.5)
    out[i, j, k] = _samp(u, _backtrace(u, v, w, x, c), wp.vec3(0.0, 0.5, 0.5))


@wp.kernel
def advect_v_k(
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32), c: float,
):  # fmt: skip
    i, j, k = wp.tid()
    x = wp.vec3(float(i) + 0.5, float(j), float(k) + 0.5)
    out[i, j, k] = _samp(v, _backtrace(u, v, w, x, c), wp.vec3(0.5, 0.0, 0.5))


@wp.kernel
def advect_w_k(
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32), c: float,
):  # fmt: skip
    i, j, k = wp.tid()
    x = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k))
    out[i, j, k] = _samp(w, _backtrace(u, v, w, x, c), wp.vec3(0.5, 0.5, 0.0))


@wp.kernel
def advect_scalars_k(
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    t: wp.array3d(dtype=wp.float32), s: wp.array3d(dtype=wp.float32), f: wp.array3d(dtype=wp.float32),
    t_out: wp.array3d(dtype=wp.float32), s_out: wp.array3d(dtype=wp.float32), f_out: wp.array3d(dtype=wp.float32),
    c: float, f_decay: float,
):  # fmt: skip
    """Semi-Lagrangian advection of ``T_g``, ``soot`` and the flame field; ``flame`` also decays by ``f_decay``
    (= exp(-k_flame dt)) here, before this step's source is added (so a steady source converges to ``P / h^3``)."""
    i, j, k = wp.tid()
    x = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    xb = _backtrace(u, v, w, x, c)
    o = wp.vec3(0.5, 0.5, 0.5)
    t_out[i, j, k] = _samp(t, xb, o)
    s_out[i, j, k] = _samp(s, xb, o)
    f_out[i, j, k] = _samp(f, xb, o) * f_decay


@wp.kernel
def scatter_sources_k(
    pos: wp.array2d(dtype=wp.float32), heat_w: wp.array(dtype=wp.float32), soot_kg_s: wp.array(dtype=wp.float32),
    power_w: wp.array(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32), t: wp.array3d(dtype=wp.float32), s: wp.array3d(dtype=wp.float32),
    fl: wp.array3d(dtype=wp.float32),
    lost: wp.array(dtype=wp.int32), roi_min: wp.vec3, inv_h: float, dt: float, k_heat: float, k_soot: float,
    k_flame: float, g: int,
):  # fmt: skip
    """Trilinear scatter at element sample points onto the 8 surrounding cell centres, fluid cells only
    (weights renormalised). ``k_heat = 1 / (rho_g c_p h^3)`` [K/J], ``k_soot = 1 / h^3`` [1/m^3],
    ``k_flame = (1 - exp(-k_flame dt)) / h^3`` [1/m^3]; the flame field gains ``power_w * k_flame`` [W/m^3]."""
    e = wp.tid()
    q = wp.vec3(pos[e, 0], pos[e, 1], pos[e, 2])
    ci = (q - roi_min) * inv_h - wp.vec3(0.5, 0.5, 0.5)
    hj = heat_w[e] * dt
    sj = soot_kg_s[e] * dt
    pj = power_w[e]
    if hj == 0.0 and sj == 0.0 and pj == 0.0:
        return
    if ci[0] < -0.5 or ci[1] < -0.5 or ci[2] < -0.5 or ci[0] > float(g) - 0.5 or ci[1] > float(g) - 0.5 or ci[2] > float(g) - 0.5:
        wp.atomic_add(lost, 0, 1)
        return
    i0 = int(wp.floor(ci[0]))
    j0 = int(wp.floor(ci[1]))
    k0 = int(wp.floor(ci[2]))
    fx = ci[0] - float(i0)
    fy = ci[1] - float(j0)
    fz = ci[2] - float(k0)
    wsum = float(0.0)  # noqa: UP018
    for c in range(8):
        dx = c & 1
        dy = (c >> 1) & 1
        dz = (c >> 2) & 1
        ii = wp.clamp(i0 + dx, 0, g - 1)
        jj = wp.clamp(j0 + dy, 0, g - 1)
        kk = wp.clamp(k0 + dz, 0, g - 1)
        wt = wp.where(dx == 1, fx, 1.0 - fx) * wp.where(dy == 1, fy, 1.0 - fy) * wp.where(dz == 1, fz, 1.0 - fz)
        if solid[ii, jj, kk] == 0:
            wsum += wt
    if wsum <= 1.0e-6:
        wp.atomic_add(lost, 0, 1)
        return
    for c in range(8):
        dx = c & 1
        dy = (c >> 1) & 1
        dz = (c >> 2) & 1
        ii = wp.clamp(i0 + dx, 0, g - 1)
        jj = wp.clamp(j0 + dy, 0, g - 1)
        kk = wp.clamp(k0 + dz, 0, g - 1)
        wt = wp.where(dx == 1, fx, 1.0 - fx) * wp.where(dy == 1, fy, 1.0 - fy) * wp.where(dz == 1, fz, 1.0 - fz)
        if solid[ii, jj, kk] == 0:
            wp.atomic_add(t, ii, jj, kk, wt / wsum * hj * k_heat)
            wp.atomic_add(s, ii, jj, kk, wt / wsum * sj * k_soot)
            wp.atomic_add(fl, ii, jj, kk, wt / wsum * pj * k_flame)


@wp.kernel
def sample_t_k(
    t: wp.array3d(dtype=wp.float32), pos: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32),
    roi_min: wp.vec3, inv_h: float,
):  # fmt: skip
    e = wp.tid()
    q = (wp.vec3(pos[e, 0], pos[e, 1], pos[e, 2]) - roi_min) * inv_h
    out[e] = _samp(t, q, wp.vec3(0.5, 0.5, 0.5))


# Single-writer face updates below use `a[i] = a[i] + x`, not `a[i] += x`: in Warp `+=` on an array is an atomic,
# and RUN_TO_RUN determinism wraps every atomic kernel in memsets and blocking device-to-host copies.
@wp.kernel
def buoyancy_k(
    w: wp.array3d(dtype=wp.float32), t: wp.array3d(dtype=wp.float32), fw: wp.array3d(dtype=wp.int32),
    dt: float, g_acc: float, t_amb: float,
):  # fmt: skip
    i, j, k = wp.tid()
    if fw[i, j, k] != 0:
        return
    n = w.shape[2] - 1
    tf = float(0.0)  # noqa: UP018
    if k >= n:
        tf = t[i, j, n - 1]
    else:
        tf = 0.5 * (t[i, j, k - 1] + t[i, j, k])
    w[i, j, k] = w[i, j, k] + (dt * g_acc * (tf - t_amb) / t_amb)


@wp.kernel
def cell_velocity_k(
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.vec3),
):  # fmt: skip
    i, j, k = wp.tid()
    out[i, j, k] = wp.vec3(
        0.5 * (u[i, j, k] + u[i + 1, j, k]), 0.5 * (v[i, j, k] + v[i, j + 1, k]), 0.5 * (w[i, j, k] + w[i, j, k + 1])
    )


@wp.kernel
def vorticity_k(
    cv: wp.array3d(dtype=wp.vec3), om: wp.array3d(dtype=wp.vec3), omm: wp.array3d(dtype=wp.float32), inv_h: float,
):  # fmt: skip
    i, j, k = wp.tid()
    g = cv.shape[0]
    ip = wp.min(i + 1, g - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, g - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, g - 1)
    km = wp.max(k - 1, 0)
    sx = inv_h / float(ip - im)
    sy = inv_h / float(jp - jm)
    sz = inv_h / float(kp - km)
    dvdx = (cv[ip, j, k][1] - cv[im, j, k][1]) * sx
    dwdx = (cv[ip, j, k][2] - cv[im, j, k][2]) * sx
    dudy = (cv[i, jp, k][0] - cv[i, jm, k][0]) * sy
    dwdy = (cv[i, jp, k][2] - cv[i, jm, k][2]) * sy
    dudz = (cv[i, j, kp][0] - cv[i, j, km][0]) * sz
    dvdz = (cv[i, j, kp][1] - cv[i, j, km][1]) * sz
    o = wp.vec3(dwdy - dvdz, dudz - dwdx, dvdx - dudy)
    om[i, j, k] = o
    omm[i, j, k] = wp.length(o)


@wp.kernel
def confinement_force_k(
    om: wp.array3d(dtype=wp.vec3), omm: wp.array3d(dtype=wp.float32), solid: wp.array3d(dtype=wp.int32),
    f: wp.array3d(dtype=wp.vec3), eps_h: float, inv_h: float,
):  # fmt: skip
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        f[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        return
    g = om.shape[0]
    ip = wp.min(i + 1, g - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, g - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, g - 1)
    km = wp.max(k - 1, 0)
    gr = wp.vec3(
        (omm[ip, j, k] - omm[im, j, k]) * inv_h / float(ip - im),
        (omm[i, jp, k] - omm[i, jm, k]) * inv_h / float(jp - jm),
        (omm[i, j, kp] - omm[i, j, km]) * inv_h / float(kp - km),
    )
    n = gr / (wp.length(gr) + 1.0e-6)
    f[i, j, k] = eps_h * wp.cross(n, om[i, j, k])


@wp.kernel
def apply_force_u_k(u: wp.array3d(dtype=wp.float32), f: wp.array3d(dtype=wp.vec3), fu: wp.array3d(dtype=wp.int32), dt: float):
    i, j, k = wp.tid()
    if fu[i, j, k] == 0:
        u[i, j, k] = u[i, j, k] + (dt * 0.5 * (f[i - 1, j, k][0] + f[i, j, k][0]))


@wp.kernel
def apply_force_v_k(v: wp.array3d(dtype=wp.float32), f: wp.array3d(dtype=wp.vec3), fv: wp.array3d(dtype=wp.int32), dt: float):
    i, j, k = wp.tid()
    if fv[i, j, k] == 0:
        v[i, j, k] = v[i, j, k] + (dt * 0.5 * (f[i, j - 1, k][1] + f[i, j, k][1]))


@wp.kernel
def apply_force_w_k(w: wp.array3d(dtype=wp.float32), f: wp.array3d(dtype=wp.vec3), fw: wp.array3d(dtype=wp.int32), dt: float):
    i, j, k = wp.tid()
    if fw[i, j, k] == 0:
        n = w.shape[2] - 1
        if k >= n:
            w[i, j, k] = w[i, j, k] + (dt * f[i, j, n - 1][2])
        else:
            w[i, j, k] = w[i, j, k] + (dt * 0.5 * (f[i, j, k - 1][2] + f[i, j, k][2]))


@wp.kernel
def clamp_face_k(a: wp.array3d(dtype=wp.float32), vmax: float, count: wp.array(dtype=wp.int32)):
    i, j, k = wp.tid()
    x = a[i, j, k]
    if wp.abs(x) > vmax:
        a[i, j, k] = wp.clamp(x, -vmax, vmax)
        wp.atomic_add(count, 0, 1)


@wp.kernel
def enforce_u_k(
    u: wp.array3d(dtype=wp.float32), fu: wp.array3d(dtype=wp.int32), solid: wp.array3d(dtype=wp.int32),
    inflow: wp.array2d(dtype=wp.float32),
):  # fmt: skip
    i, j, k = wp.tid()
    if fu[i, j, k] != 0:
        val = float(0.0)  # noqa: UP018
        if i == 0 and solid[0, j, k] == 0:
            val = inflow[j, k]
        u[i, j, k] = val


@wp.kernel
def enforce_k(a: wp.array3d(dtype=wp.float32), fixed: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    if fixed[i, j, k] != 0:
        a[i, j, k] = 0.0


@wp.kernel
def divergence_k(
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32), div: wp.array3d(dtype=wp.float32), inv_h: float,
):  # fmt: skip
    # no norm accumulation here: one global atomic per cell dominated the step under RUN_TO_RUN determinism;
    # divergence_norms() reduces the stored field on demand instead
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        div[i, j, k] = 0.0
        return
    div[i, j, k] = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) * inv_h


@wp.kernel
def rbgs_k(
    p: wp.array3d(dtype=wp.float32), div: wp.array3d(dtype=wp.float32), solid: wp.array3d(dtype=wp.int32),
    fu: wp.array3d(dtype=wp.int32), fv: wp.array3d(dtype=wp.int32), fw: wp.array3d(dtype=wp.int32),
    color: int, h2_dt: float,
):  # fmt: skip
    """One half-sweep of red-black Gauss-Seidel on ``sum_free (p_c - p_nb) = -(h^2/dt) div`` (Neumann at
    fixed faces, ghost ``p = 0`` above the open top face)."""
    i, j, kk = wp.tid()  # launched over (G, G, G/2): one thread per cell of the given colour
    k = 2 * kk + ((i + j + color) & 1)
    if solid[i, j, k] != 0:
        return
    g = p.shape[0]
    n = float(0.0)  # noqa: UP018
    s = float(0.0)  # noqa: UP018
    if fu[i, j, k] == 0:
        n += 1.0
        s += p[i - 1, j, k]
    if fu[i + 1, j, k] == 0:
        n += 1.0
        s += p[i + 1, j, k]
    if fv[i, j, k] == 0:
        n += 1.0
        s += p[i, j - 1, k]
    if fv[i, j + 1, k] == 0:
        n += 1.0
        s += p[i, j + 1, k]
    if fw[i, j, k] == 0:
        n += 1.0
        s += p[i, j, k - 1]
    if fw[i, j, k + 1] == 0:
        n += 1.0
        if k + 1 < g:
            s += p[i, j, k + 1]
    if n > 0.0:
        p[i, j, k] = (s - h2_dt * div[i, j, k]) / n
    else:
        p[i, j, k] = 0.0


def rbgs_mask(solid: np.ndarray, fu: np.ndarray, fv: np.ndarray, fw: np.ndarray) -> np.ndarray:
    """Per-cell uint8 for ``rbgs_mask_k``: bits 0..5 = face open (flag 0) at -x, +x, -y, +y, -z, +z; bit 6 = solid.
    Replaces seven int32 reads per cell with one byte (the face flags are static)."""
    m = ((fu[:-1] == 0) * 1 + (fu[1:] == 0) * 2 + (fv[:, :-1] == 0) * 4 + (fv[:, 1:] == 0) * 8
         + (fw[:, :, :-1] == 0) * 16 + (fw[:, :, 1:] == 0) * 32 + (solid != 0) * 64)  # fmt: skip
    return m.astype(np.uint8)


@wp.kernel
def rbgs_mask_k(
    p: wp.array3d(dtype=wp.float32), div: wp.array3d(dtype=wp.float32), mask: wp.array3d(dtype=wp.uint8),
    color: int, h2_dt: float,
):  # fmt: skip
    """``rbgs_k`` with the face flags packed by :func:`rbgs_mask`; same summation order, so bit-identical."""
    i, j, kk = wp.tid()
    k = 2 * kk + ((i + j + color) & 1)
    m = int(mask[i, j, k])
    if (m & 64) != 0:
        return
    g = p.shape[0]
    n = float(0.0)  # noqa: UP018
    s = float(0.0)  # noqa: UP018
    if (m & 1) != 0:
        n += 1.0
        s += p[i - 1, j, k]
    if (m & 2) != 0:
        n += 1.0
        s += p[i + 1, j, k]
    if (m & 4) != 0:
        n += 1.0
        s += p[i, j - 1, k]
    if (m & 8) != 0:
        n += 1.0
        s += p[i, j + 1, k]
    if (m & 16) != 0:
        n += 1.0
        s += p[i, j, k - 1]
    if (m & 32) != 0:
        n += 1.0
        if k + 1 < g:
            s += p[i, j, k + 1]
    if n > 0.0:
        p[i, j, k] = (s - h2_dt * div[i, j, k]) / n
    else:
        p[i, j, k] = 0.0


@wp.kernel
def grad_u_k(u: wp.array3d(dtype=wp.float32), p: wp.array3d(dtype=wp.float32), fu: wp.array3d(dtype=wp.int32), c: float):
    i, j, k = wp.tid()
    if fu[i, j, k] == 0:
        u[i, j, k] = u[i, j, k] - (c * (p[i, j, k] - p[i - 1, j, k]))


@wp.kernel
def grad_v_k(v: wp.array3d(dtype=wp.float32), p: wp.array3d(dtype=wp.float32), fv: wp.array3d(dtype=wp.int32), c: float):
    i, j, k = wp.tid()
    if fv[i, j, k] == 0:
        v[i, j, k] = v[i, j, k] - (c * (p[i, j, k] - p[i, j - 1, k]))


@wp.kernel
def grad_w_k(w: wp.array3d(dtype=wp.float32), p: wp.array3d(dtype=wp.float32), fw: wp.array3d(dtype=wp.int32), c: float):
    i, j, k = wp.tid()
    if fw[i, j, k] == 0:
        pb = float(0.0)  # noqa: UP018
        if k < p.shape[2]:
            pb = p[i, j, k]
        w[i, j, k] = w[i, j, k] - (c * (pb - p[i, j, k - 1]))


@wp.kernel
def finish_k(
    t: wp.array3d(dtype=wp.float32), s: wp.array3d(dtype=wp.float32), fl: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    cool: float, decay: float, t_amb: float, t_max: float, count: wp.array(dtype=wp.int32),
):  # fmt: skip
    i, j, k = wp.tid()
    if solid[i, j, k] != 0:
        t[i, j, k] = t_amb
        s[i, j, k] = 0.0
        fl[i, j, k] = 0.0
        return
    tn = t_amb + (t[i, j, k] - t_amb) * cool
    if tn > t_max:
        tn = t_max
        wp.atomic_add(count, 0, 1)
    t[i, j, k] = tn
    s[i, j, k] = wp.max(s[i, j, k] * decay, 0.0)


# --------------------------------------------------------------------------------------------------
class GasSolver:
    """MAC-grid gas solver on ``G^3`` cells of size ``h`` [m].

    Owns float32 device arrays: ``u (G+1,G,G)``, ``v (G,G+1,G)``, ``w (G,G,G+1)`` [m/s] face velocities,
    ``T_g (G,G,G)`` [K], ``soot (G,G,G)`` [kg/m^3, mass concentration], ``flame (G,G,G)`` [W/m^3] combustion-intensity
    (heat-release-rate density proxy, a passive scalar for the renderer only, ADR-0004), ``p (G,G,G)`` kinematic
    pressure [m^2/s^2] (multiply by ``rho_g`` for Pa). ``solid_mask`` (G,G,G) uint8, host. World frame, index
    ``[ix,iy,iz]``, cell centre ``roi_min + (i+.5) h``. Parameters from ``configs/sim.yaml`` (``Params``).
    Vorticity confinement is applied as ``f = eps_vc * h * (N x omega)`` [m/s^2] (N = normalised gradient of
    ``|omega|``), i.e. an artistic, grid-scaled strength per unit time, not a physical coefficient.
    Single owner, not thread-safe. On CPU all kernels (including source scatter) are deterministic.
    """

    def __init__(
        self,
        G: int,
        h: float,
        roi_min: np.ndarray,
        solid_mask: np.ndarray,
        params: Params,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        self.device = select_device(device)
        d = self.device
        self.G, self.h = int(G), float(h)
        self.roi_min = wp.vec3(*[float(x) for x in np.asarray(roi_min, dtype=np.float64)])
        if G % 2:
            raise ValueError("G must be even (red-black half-sweeps)")
        solid = np.asarray(solid_mask).astype(np.int32)
        if solid.shape != (G, G, G):
            raise ValueError(f"solid_mask must be ({G},{G},{G}), got {solid.shape}")
        self._p = {k: params[k] for k in params.keys()}  # noqa: SIM118
        self._rng = np.random.default_rng(seed)
        self.solid = to_device(solid, wp.int32, d)
        # fixed (no-through) face flags: walls and faces touching a solid cell. Top face (z=roi_max) is free.
        fu = np.zeros((G + 1, G, G), np.int32)
        fv = np.zeros((G, G + 1, G), np.int32)
        fw = np.zeros((G, G, G + 1), np.int32)
        fu[0], fu[G] = 1, 1
        fv[:, 0], fv[:, G] = 1, 1
        fw[:, :, 0] = 1
        fu[1:G] |= solid[:-1] | solid[1:]
        fv[:, 1:G] |= solid[:, :-1] | solid[:, 1:]
        fw[:, :, 1:G] |= solid[:, :, :-1] | solid[:, :, 1:]
        fw[:, :, G] = solid[:, :, G - 1]  # open top face is blocked only if its cell is solid
        self.fu, self.fv, self.fw = (to_device(a, wp.int32, d) for a in (fu, fv, fw))
        self._rb_mask = to_device(rbgs_mask(solid, fu, fv, fw), wp.uint8, d)  # static: built from the flags above
        z3 = lambda *s: wp.zeros(s, dtype=wp.float32, device=d)
        self.u, self.v, self.w = z3(G + 1, G, G), z3(G, G + 1, G), z3(G, G, G + 1)
        self._u2, self._v2, self._w2 = z3(G + 1, G, G), z3(G, G + 1, G), z3(G, G, G + 1)
        self.T_g = to_device(np.full((G, G, G), self._p["T_amb"], np.float32), wp.float32, d)
        self._T2 = z3(G, G, G)
        self.soot = z3(G, G, G)
        self._s2 = z3(G, G, G)
        self.flame = z3(G, G, G)
        self._f2 = z3(G, G, G)
        self._no_power = z3(1)  # zeros stand-in (resized on demand) for steps without released-power input
        self.p = z3(G, G, G)
        self._div = z3(G, G, G)  # divergence before the last projection (RBGS right-hand side)
        self._div_after = z3(G, G, G)  # divergence after the last projection
        self._cv = wp.zeros((G, G, G), dtype=wp.vec3, device=d)
        self._om = wp.zeros((G, G, G), dtype=wp.vec3, device=d)
        self._fc = wp.zeros((G, G, G), dtype=wp.vec3, device=d)
        self._omm = z3(G, G, G)
        self._inflow = wp.zeros((G, G), dtype=wp.float32, device=d)
        self._vel_clamps = wp.zeros(1, dtype=wp.int32, device=d)
        self._T_clamps = wp.zeros(1, dtype=wp.int32, device=d)
        self._lost = wp.zeros(1, dtype=wp.int32, device=d)
        self.step_count = 0
        self.reset()

    # ---- state ------------------------------------------------------------------------------
    def reset(self) -> None:
        """Quiescent ambient state; counters cleared."""
        for a in (self.u, self.v, self.w, self.soot, self.flame, self.p, self._div, self._div_after):
            a.zero_()
        self.T_g.fill_(self._p["T_amb"])
        for c in (self._vel_clamps, self._T_clamps, self._lost):
            c.zero_()
        self.step_count = 0

    def set_velocity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        """Load face velocities from host float arrays (shapes (G+1,G,G), (G,G+1,G), (G,G,G+1)); no BC applied."""
        self.u.assign(np.ascontiguousarray(u, dtype=np.float32))
        self.v.assign(np.ascontiguousarray(v, dtype=np.float32))
        self.w.assign(np.ascontiguousarray(w, dtype=np.float32))

    def set_scalars(
        self, T_g: np.ndarray | None = None, soot: np.ndarray | None = None, flame: np.ndarray | None = None
    ) -> None:
        """Load cell-centred ``T_g`` [K], ``soot`` and/or ``flame`` [W/m^3] (G,G,G) from host arrays."""
        if flame is not None:
            self.flame.assign(np.ascontiguousarray(flame, dtype=np.float32))
        if T_g is not None:
            self.T_g.assign(np.ascontiguousarray(T_g, dtype=np.float32))
        if soot is not None:
            self.soot.assign(np.ascontiguousarray(soot, dtype=np.float32))

    # ---- sub-steps (public for tests/benchmarks) --------------------------------------------
    def enforce_boundaries(self) -> None:
        """Set fixed faces (walls, solid-touching) to 0, except the -x wall which carries the wind inflow."""
        d = self.device
        wp.launch(enforce_u_k, dim=self.u.shape, inputs=[self.u, self.fu, self.solid, self._inflow], device=d)
        wp.launch(enforce_k, dim=self.v.shape, inputs=[self.v, self.fv], device=d)
        wp.launch(enforce_k, dim=self.w.shape, inputs=[self.w, self.fw], device=d)

    def _divergence(self, out: wp.array) -> None:
        wp.launch(divergence_k, dim=(self.G,) * 3, device=self.device,
                  inputs=[self.u, self.v, self.w, self.solid, out, 1.0 / self.h])  # fmt: skip

    def project(self, dt: float, n_iters: int | None = None) -> tuple[float, float]:
        """Pressure projection (``n_iters`` red-black GS iterations, warm-started from the previous ``p``).
        Returns ``(||div||_2 before, after)`` over fluid cells [1/s] (syncs; for tests)."""
        self._project(dt, int(self._p["n_iters"]) if n_iters is None else int(n_iters))
        return self.divergence_norms()

    def _project(self, dt: float, n_iters: int) -> None:
        d, G, h = self.device, self.G, self.h
        self.enforce_boundaries()
        self._divergence(self._div)
        h2_dt = h * h / dt
        for _ in range(n_iters):
            for color in (0, 1):
                wp.launch(rbgs_mask_k, dim=(G, G, G // 2), device=d,
                          inputs=[self.p, self._div, self._rb_mask, color, h2_dt])  # fmt: skip
        c = dt / h
        wp.launch(grad_u_k, dim=self.u.shape, inputs=[self.u, self.p, self.fu, c], device=d)
        wp.launch(grad_v_k, dim=self.v.shape, inputs=[self.v, self.p, self.fv, c], device=d)
        wp.launch(grad_w_k, dim=self.w.shape, inputs=[self.w, self.p, self.fw, c], device=d)
        self._divergence(self._div_after)

    def divergence_norms(self) -> tuple[float, float]:
        """``(before, after)`` ``||div u||_2`` of the last projection (syncs)."""
        b, a = (float(np.sqrt(np.square(x.numpy(), dtype=np.float64).sum())) for x in (self._div, self._div_after))
        return b, a

    # ---- main step --------------------------------------------------------------------------
    def step(
        self,
        dt: float,
        src_pos: wp.array | None = None,
        heat_W: wp.array | None = None,
        soot_kg_s: wp.array | None = None,
        power_W: wp.array | None = None,
    ) -> None:
        """Advance ``dt`` [s]. Optional element sources: ``src_pos`` (N,3) float32 world [m], ``heat_W`` (N,)
        [W], ``soot_kg_s`` (N,) [kg/s] (device arrays); injected as ``heat_J = heat_W dt``, ``soot_kg = soot_kg_s dt``.
        ``power_W`` (N,) [W] is the total released power per element (feedback + gas parts); it feeds the
        ``flame`` field as ``flame += P / h^3 (1 - exp(-k_flame dt))`` (None: no flame source)."""
        p, d, G, h = self._p, self.device, self.G, self.h
        c = dt / h
        dim = (G,) * 3
        # 1. advection
        wp.launch(advect_u_k, dim=self.u.shape, inputs=[self.u, self.v, self.w, self._u2, c], device=d)
        wp.launch(advect_v_k, dim=self.v.shape, inputs=[self.u, self.v, self.w, self._v2, c], device=d)
        wp.launch(advect_w_k, dim=self.w.shape, inputs=[self.u, self.v, self.w, self._w2, c], device=d)
        wp.launch(
            advect_scalars_k, dim=dim, device=d,
            inputs=[self.u, self.v, self.w, self.T_g, self.soot, self.flame, self._T2, self._s2, self._f2, c,
                    math.exp(-p["k_flame"] * dt)],
        )  # fmt: skip
        self.u, self._u2 = self._u2, self.u
        self.v, self._v2 = self._v2, self.v
        self.w, self._w2 = self._w2, self.w
        self.T_g, self._T2 = self._T2, self.T_g
        self.soot, self._s2 = self._s2, self.soot
        self.flame, self._f2 = self._f2, self.flame
        # 2. sources
        if src_pos is not None and heat_W is not None and soot_kg_s is not None:
            k_heat = 1.0 / (p["rho_g"] * p["c_p"] * h**3)
            k_fl = (1.0 - math.exp(-p["k_flame"] * dt)) / h**3
            if power_W is None:
                if self._no_power.shape[0] != src_pos.shape[0]:
                    self._no_power = wp.zeros(src_pos.shape[0], dtype=wp.float32, device=d)
                pw = self._no_power
            else:
                pw = power_W
            wp.launch(
                scatter_sources_k, dim=src_pos.shape[0], device=d,
                inputs=[src_pos, heat_W, soot_kg_s, pw, self.solid, self.T_g, self.soot, self.flame, self._lost,
                        self.roi_min, 1.0 / h, dt, k_heat, 1.0 / h**3, k_fl, G],
            )  # fmt: skip
        # 3. buoyancy
        wp.launch(buoyancy_k, dim=self.w.shape, inputs=[self.w, self.T_g, self.fw, dt, p["g"], p["T_amb"]], device=d)
        # 4. vorticity confinement
        if p["eps_vc"] > 0.0:
            self._confinement(dt)
        # 5. wind (inflow on the -x wall, seeded noise)
        if p["wind_speed"] > 0.0 or p["wind_noise"] > 0.0:
            inflow = p["wind_speed"] + p["wind_noise"] * self._rng.standard_normal((G, G))
            self._inflow.assign(inflow.astype(np.float32))
        # 6. velocity clamp |u| <= cfl_max h / dt, counted
        vmax = p["cfl_max"] * h / dt
        for a in (self.u, self.v, self.w):
            wp.launch(clamp_face_k, dim=a.shape, inputs=[a, vmax, self._vel_clamps], device=d)
        # 7. projection
        self._project(dt, int(p["n_iters"]))
        # 8. cooling, soot decay, temperature clamp
        wp.launch(
            finish_k, dim=dim, device=d,
            inputs=[self.T_g, self.soot, self.flame, self.solid, math.exp(-p["k_cool"] * dt), math.exp(-p["k_soot"] * dt),
                    p["T_amb"], p["T_g_max"], self._T_clamps],
        )  # fmt: skip
        self.step_count += 1

    def _confinement(self, dt: float) -> None:
        d, G, h = self.device, self.G, self.h
        dim = (G,) * 3
        wp.launch(cell_velocity_k, dim=dim, inputs=[self.u, self.v, self.w, self._cv], device=d)
        wp.launch(vorticity_k, dim=dim, inputs=[self._cv, self._om, self._omm, 1.0 / h], device=d)
        wp.launch(
            confinement_force_k, dim=dim, device=d,
            inputs=[self._om, self._omm, self.solid, self._fc, self._p["eps_vc"] * h, 1.0 / h],
        )  # fmt: skip
        wp.launch(apply_force_u_k, dim=self.u.shape, inputs=[self.u, self._fc, self.fu, dt], device=d)
        wp.launch(apply_force_v_k, dim=self.v.shape, inputs=[self.v, self._fc, self.fv, dt], device=d)
        wp.launch(apply_force_w_k, dim=self.w.shape, inputs=[self.w, self._fc, self.fw, dt], device=d)

    # ---- sampling / readback ----------------------------------------------------------------
    def sample_T(self, pos: wp.array, out: wp.array) -> None:
        """Trilinear ``T_g`` [K] at world points ``pos`` (N,3) float32 into ``out`` (N,) float32 (device)."""
        wp.launch(
            sample_t_k, dim=pos.shape[0], inputs=[self.T_g, pos, out, self.roi_min, 1.0 / self.h], device=self.device
        )

    def velocity_cell_centered(self) -> np.ndarray:
        """Cell-centred velocity ``(G,G,G,3)`` float32 [m/s], world axes (average of the two bounding faces).
        Host copy; the caller owns it (ADR-0002)."""
        wp.launch(cell_velocity_k, dim=(self.G,) * 3, inputs=[self.u, self.v, self.w, self._cv], device=self.device)
        wp.synchronize_device(self.device)
        return self._cv.numpy().copy()

    def counters(self) -> dict[str, int]:
        """Cumulative counters since :meth:`reset` (syncs): velocity clamp violations (face events), ``T_g``
        clamp events, source scatters dropped (outside ROI or only solid neighbours)."""
        wp.synchronize_device(self.device)
        return {
            "vel_clamp": int(self._vel_clamps.numpy()[0]),
            "T_g_clamp": int(self._T_clamps.numpy()[0]),
            "src_lost": int(self._lost.numpy()[0]),
        }
