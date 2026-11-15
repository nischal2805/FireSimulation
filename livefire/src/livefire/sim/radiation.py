"""Radiative heat transfer between surface elements (P2-001, PRD 8.2, P2R2/P2R4/P2R5).

Burning elements are emitters with temperature ``T_b = max(T_s_b, T_rad_flame)`` (the attached flame) and emissive
power ``w_b = eps_f sigma T_b^4 A_b`` [W]. The absorbed flux at receiver ``i`` is

    q_rad_i = eps_i * sum_b w_b c_i c_b / (pi max(r, r_min)^2),   r = |x_b - x_i| <= R,

with ``c_b = |cos th_b|`` (flames envelop the element), ``c_i = |cos th_i|`` for thin two-sided receivers (material
``consumed_when_burnt``) and ``max(cos th_i, 0)`` otherwise; the self term is excluded. No visibility test (PRD 8.2
option 4 is deferred).

Mode 0 writes zeros. Mode 2 compacts the emitters in element-index order (``wp.utils.array_scan`` over the burning
flags: no atomic-counter ordering) and gathers per receiver over the list in list order. Mode 1 bins emitter power on
its own grid of ``rad_cell`` metres over the ROI (independent of the gas grid) keeping per cell the power ``W``, the
power-weighted centroid and the power-weighted tensor ``M = sum w n n^T`` (so ``c_b^2 ~ d.M.d / (W r^2)``: the
sign-invariant analogue of a "mean normal"), compacts active cells the same way and gathers over them. Cell binning
uses ``wp.atomic_add`` (deterministic through ``wp.config.deterministic``; CPU is serial and exact).

All device arrays are float64 except flags/indices (int32). The model owns its arrays; the host inputs are copied.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from livefire.common.config import ConfigError, Params
from livefire.common.geometry import AABB
from livefire.sim.device import select_device, to_device

SIGMA = 5.670374419e-8  # W/(m^2 K^4)
_F64 = wp.float64
MAX_CELLS = 2_000_000  # mode 1 grid cap (10 float64 arrays per cell)


@wp.kernel
def _emission_k(
    burning: wp.array(dtype=wp.uint8),
    t_s: wp.array(dtype=wp.float64),
    area: wp.array(dtype=wp.float64),
    eps_f: wp.float64,
    t_flame: wp.float64,
    w_all: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    sum_w: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    w = _F64(0.0)
    f = wp.int32(0)
    if burning[i] != wp.uint8(0):
        tb = wp.max(t_s[i], t_flame)
        w = eps_f * _F64(SIGMA) * tb * tb * tb * tb * area[i]
        f = 1
        wp.atomic_add(sum_w, 0, w)
    w_all[i] = w
    flags[i] = f


@wp.kernel
def _compact_k(
    flags: wp.array(dtype=wp.int32),
    offs: wp.array(dtype=wp.int32),
    cap: int,
    lst: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if flags[i] != 0:
        k = offs[i] - 1  # inclusive scan
        if k < cap:
            lst[k] = i


@wp.kernel
def _finalize_k(
    offs: wp.array(dtype=wp.int32),
    n_flags: int,
    cap: int,
    count: wp.array(dtype=wp.int32),
    total: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int64),
):
    t = offs[n_flags - 1]
    total[0] = t
    c = wp.min(t, cap)
    count[0] = c
    overflow[0] = overflow[0] + wp.int64(t - c)


@wp.kernel
def _gather_elements_k(
    pos: wp.array2d(dtype=wp.float64),
    nrm: wp.array2d(dtype=wp.float64),
    eps: wp.array(dtype=wp.float64),
    two_sided: wp.array(dtype=wp.uint8),
    w_all: wp.array(dtype=wp.float64),
    lst: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    r_max2: wp.float64,
    r_min2: wp.float64,
    out_q: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    n_e = count[0]
    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    nx = nrm[i, 0]
    ny = nrm[i, 1]
    nz = nrm[i, 2]
    acc = _F64(0.0)
    for e in range(n_e):
        j = lst[e]
        if j == i:
            continue
        dx = pos[j, 0] - px
        dy = pos[j, 1] - py
        dz = pos[j, 2] - pz
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > r_max2 or r2 <= _F64(1.0e-18):  # coincident points: direction undefined (round-off noise)
            continue
        r = wp.sqrt(r2)
        ci = (nx * dx + ny * dy + nz * dz) / r
        if two_sided[i] != wp.uint8(0):
            ci = wp.abs(ci)
        else:
            ci = wp.max(ci, _F64(0.0))
        cb = wp.abs(nrm[j, 0] * dx + nrm[j, 1] * dy + nrm[j, 2] * dz) / r
        acc += w_all[j] * ci * cb / (_F64(math.pi) * wp.max(r2, r_min2))
    out_q[i] = eps[i] * acc


@wp.kernel
def _bin_k(
    w_all: wp.array(dtype=wp.float64),
    pos: wp.array2d(dtype=wp.float64),
    nrm: wp.array2d(dtype=wp.float64),
    lo: wp.vec3d,
    inv_cell: wp.float64,
    dims: wp.vec3i,
    cw: wp.array(dtype=wp.float64),
    cx: wp.array2d(dtype=wp.float64),
    cm: wp.array2d(dtype=wp.float64),
):
    i = wp.tid()
    w = w_all[i]
    if w <= _F64(0.0):
        return
    c = _cell_of(pos[i, 0], pos[i, 1], pos[i, 2], lo, inv_cell, dims)
    wp.atomic_add(cw, c, w)
    for a in range(3):
        wp.atomic_add(cx, c, a, w * pos[i, a])
    nx = nrm[i, 0]
    ny = nrm[i, 1]
    nz = nrm[i, 2]
    wp.atomic_add(cm, c, 0, w * nx * nx)
    wp.atomic_add(cm, c, 1, w * ny * ny)
    wp.atomic_add(cm, c, 2, w * nz * nz)
    wp.atomic_add(cm, c, 3, w * nx * ny)
    wp.atomic_add(cm, c, 4, w * nx * nz)
    wp.atomic_add(cm, c, 5, w * ny * nz)


@wp.func
def _cell_of(x: wp.float64, y: wp.float64, z: wp.float64, lo: wp.vec3d, inv_cell: wp.float64, dims: wp.vec3i) -> int:
    ix = wp.clamp(int((x - lo[0]) * inv_cell), 0, dims[0] - 1)
    iy = wp.clamp(int((y - lo[1]) * inv_cell), 0, dims[1] - 1)
    iz = wp.clamp(int((z - lo[2]) * inv_cell), 0, dims[2] - 1)
    return (iz * dims[1] + iy) * dims[0] + ix


@wp.kernel
def _cell_flags_k(cw: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if cw[i] > _F64(0.0):
        flags[i] = 1
    else:
        flags[i] = 0


@wp.kernel
def _gather_cells_k(
    pos: wp.array2d(dtype=wp.float64),
    nrm: wp.array2d(dtype=wp.float64),
    eps: wp.array(dtype=wp.float64),
    two_sided: wp.array(dtype=wp.uint8),
    w_all: wp.array(dtype=wp.float64),
    lo: wp.vec3d,
    inv_cell: wp.float64,
    dims: wp.vec3i,
    cw: wp.array(dtype=wp.float64),
    cx: wp.array2d(dtype=wp.float64),
    cm: wp.array2d(dtype=wp.float64),
    lst: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    r_max2: wp.float64,
    r_min2: wp.float64,
    out_q: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    n_c = count[0]
    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    nx = nrm[i, 0]
    ny = nrm[i, 1]
    nz = nrm[i, 2]
    wi = w_all[i]  # own emission (0 if not burning): removed from the receiver's own cell (self term excluded)
    own = _cell_of(px, py, pz, lo, inv_cell, dims)
    acc = _F64(0.0)
    for e in range(n_c):
        c = lst[e]
        w = cw[c]
        sx = cx[c, 0]
        sy = cx[c, 1]
        sz = cx[c, 2]
        m0 = cm[c, 0]
        m1 = cm[c, 1]
        m2 = cm[c, 2]
        m3 = cm[c, 3]
        m4 = cm[c, 4]
        m5 = cm[c, 5]
        if c == own and wi > _F64(0.0):
            w -= wi
            sx -= wi * px
            sy -= wi * py
            sz -= wi * pz
            m0 -= wi * nx * nx
            m1 -= wi * ny * ny
            m2 -= wi * nz * nz
            m3 -= wi * nx * ny
            m4 -= wi * nx * nz
            m5 -= wi * ny * nz
        if w <= _F64(1.0e-9) * cw[c]:
            continue
        dx = sx / w - px
        dy = sy / w - py
        dz = sz / w - pz
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > r_max2 or r2 <= _F64(1.0e-18):  # coincident points: direction undefined (round-off noise)
            continue
        r = wp.sqrt(r2)
        ci = (nx * dx + ny * dy + nz * dz) / r
        if two_sided[i] != wp.uint8(0):
            ci = wp.abs(ci)
        else:
            ci = wp.max(ci, _F64(0.0))
        dmd = m0 * dx * dx + m1 * dy * dy + m2 * dz * dz + _F64(2.0) * (m3 * dx * dy + m4 * dx * dz + m5 * dy * dz)
        cb = wp.sqrt(wp.max(dmd, _F64(0.0)) / w) / r
        acc += w * ci * cb / (_F64(math.pi) * wp.max(r2, r_min2))
    out_q[i] = eps[i] * acc


@wp.kernel
def _absorbed_k(q: wp.array(dtype=wp.float64), area: wp.array(dtype=wp.float64), p_rx: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(p_rx, 0, q[i] * area[i])


class RadiationModel:
    """Radiative exchange for N surface elements (see module docstring).

    Host inputs (float64): ``pos`` (N,3) m world, ``normal`` (N,3) unit, ``area`` (N,) m^2, ``eps`` (N,) receiver
    absorptivity, ``two_sided`` (N,) bool (thin sheets: paper, cloth), ``roi`` AABB (mode 1 grid), ``params`` (needs
    ``rad_mode, eps_f, T_rad_flame, rad_R, rad_r_min, rad_cell, rad_max_emitters``). Owns device arrays; the caller
    owns the arrays passed to :meth:`compute`. Device = ``device``. Not thread-safe.

    Counters (device, cumulative where noted): ``overflow`` (int64, cumulative emitters/cells dropped by the
    ``rad_max_emitters`` cap), ``total`` (list length before the cap, last call), ``p_rx`` (sum A q, last call),
    ``sum_w`` (sum of emitter powers, last call), ``w_all`` (N,) emitter power [W] of the last call (0 for
    non-emitters)."""

    def __init__(
        self,
        pos: np.ndarray,
        normal: np.ndarray,
        area: np.ndarray,
        eps: np.ndarray,
        two_sided: np.ndarray,
        roi: AABB,
        params: Params,
        device: str | None = None,
    ) -> None:
        self.device = d = select_device(device)
        p = params
        self.mode = round(p["rad_mode"]) if "rad_mode" in p else 0
        if self.mode not in (0, 1, 2):
            raise ConfigError(f"rad_mode must be 0, 1 or 2, got {self.mode}")
        self.n = int(pos.shape[0])
        self.eps_f = float(p["eps_f"]) if "eps_f" in p else 0.7
        self.t_flame = float(p["T_rad_flame"]) if "T_rad_flame" in p else 1100.0
        r_max = float(p["rad_R"]) if "rad_R" in p else 0.30
        r_min = float(p["rad_r_min"]) if "rad_r_min" in p else 0.01
        self.cell = float(p["rad_cell"]) if "rad_cell" in p else 0.05
        self.cap = int(p["rad_max_emitters"]) if "rad_max_emitters" in p else 8192
        self.r_max2, self.r_min2 = r_max * r_max, r_min * r_min
        n = self.n
        self.pos = to_device(np.asarray(pos, dtype=np.float64), wp.float64, d)
        self.nrm = to_device(np.asarray(normal, dtype=np.float64), wp.float64, d)
        self.area = to_device(np.asarray(area, dtype=np.float64), wp.float64, d)
        self.eps = to_device(np.asarray(eps, dtype=np.float64), wp.float64, d)
        self.two_sided = to_device(np.asarray(two_sided, dtype=np.uint8), wp.uint8, d)
        self.w_all = wp.zeros(n, dtype=wp.float64, device=d)
        self.sum_w = wp.zeros(1, dtype=wp.float64, device=d)
        self.p_rx = wp.zeros(1, dtype=wp.float64, device=d)
        self.overflow = wp.zeros(1, dtype=wp.int64, device=d)
        self.total = wp.zeros(1, dtype=wp.int32, device=d)
        self._count = wp.zeros(1, dtype=wp.int32, device=d)
        self._flags = wp.zeros(max(n, 1), dtype=wp.int32, device=d)
        self._offs = wp.zeros(max(n, 1), dtype=wp.int32, device=d)
        self._list = wp.zeros(max(1, min(self.cap, n)), dtype=wp.int32, device=d)  # mode 2: emitter ids
        if self.mode == 1:
            lo = np.asarray(roi.min, dtype=np.float64)
            ext = np.asarray(roi.max, dtype=np.float64) - lo
            dims = np.maximum(1, np.ceil(ext / self.cell - 1e-9).astype(int))
            nc = int(dims.prod())
            if nc > MAX_CELLS:
                raise ConfigError(f"rad_cell {self.cell} m gives {nc} cells over the ROI (max {MAX_CELLS})")
            self._lo = wp.vec3d(*lo)
            self._dims = wp.vec3i(*[int(x) for x in dims])
            self._inv_cell = 1.0 / self.cell
            self._cw = wp.zeros(nc, dtype=wp.float64, device=d)
            self._cx = wp.zeros((nc, 3), dtype=wp.float64, device=d)
            self._cm = wp.zeros((nc, 6), dtype=wp.float64, device=d)
            self._cflags = wp.zeros(nc, dtype=wp.int32, device=d)
            self._coffs = wp.zeros(nc, dtype=wp.int32, device=d)
            self._list = wp.zeros(max(1, min(self.cap, nc)), dtype=wp.int32, device=d)  # mode 1: active cell ids

    def compute(self, burning: wp.array, T_s: wp.array, out_q: wp.array) -> None:
        """Fill ``out_q`` (N,) float64 [W/m^2 absorbed] from ``burning`` (N,) uint8 and ``T_s`` (N,) float64 [K]
        (device arrays on this device). Mode 0 writes zeros and launches no gather kernel. No device sync."""
        d = self.device
        self.sum_w.zero_()
        self.p_rx.zero_()
        if self.mode == 0 or self.n == 0:
            out_q.zero_()
            self.w_all.zero_()
            self.total.zero_()
            return
        wp.launch(_emission_k, dim=self.n, device=d,
                  inputs=[burning, T_s, self.area, self.eps_f, self.t_flame, self.w_all, self._flags, self.sum_w])  # fmt: skip
        if self.mode == 2:
            wp.utils.array_scan(self._flags, self._offs, inclusive=True)
            wp.launch(_compact_k, dim=self.n, device=d, inputs=[self._flags, self._offs, self.cap, self._list])
            wp.launch(_finalize_k, dim=1, device=d,
                      inputs=[self._offs, self.n, self.cap, self._count, self.total, self.overflow])  # fmt: skip
            wp.launch(_gather_elements_k, dim=self.n, device=d,
                      inputs=[self.pos, self.nrm, self.eps, self.two_sided, self.w_all, self._list, self._count,
                              self.r_max2, self.r_min2, out_q])  # fmt: skip
        else:
            self._cw.zero_()
            self._cx.zero_()
            self._cm.zero_()
            wp.launch(_bin_k, dim=self.n, device=d,
                      inputs=[self.w_all, self.pos, self.nrm, self._lo, self._inv_cell, self._dims,
                              self._cw, self._cx, self._cm])  # fmt: skip
            nc = self._cw.shape[0]
            wp.launch(_cell_flags_k, dim=nc, device=d, inputs=[self._cw, self._cflags])
            wp.utils.array_scan(self._cflags, self._coffs, inclusive=True)
            wp.launch(_compact_k, dim=nc, device=d, inputs=[self._cflags, self._coffs, self.cap, self._list])
            wp.launch(_finalize_k, dim=1, device=d,
                      inputs=[self._coffs, nc, self.cap, self._count, self.total, self.overflow])  # fmt: skip
            wp.launch(_gather_cells_k, dim=self.n, device=d,
                      inputs=[self.pos, self.nrm, self.eps, self.two_sided, self.w_all, self._lo, self._inv_cell,
                              self._dims, self._cw, self._cx, self._cm, self._list, self._count,
                              self.r_max2, self.r_min2, out_q])  # fmt: skip
        wp.launch(_absorbed_k, dim=self.n, device=d, inputs=[out_q, self.area, self.p_rx])

    # ---- readback (sync) ------------------------------------------------------------------
    @property
    def last_emitters(self) -> int:
        """Emitters (mode 2) or active cells (mode 1) found by the last :meth:`compute`, before the cap (syncs)."""
        return int(self.total.numpy()[0])

    @property
    def overflow_count(self) -> int:
        """Cumulative number of emitters / cells dropped because of ``rad_max_emitters`` (syncs)."""
        return int(self.overflow.numpy()[0])

    @property
    def last_absorbed_W(self) -> float:
        """``sum_i A_i q_rad_i`` [W] of the last :meth:`compute` (syncs)."""
        return float(self.p_rx.numpy()[0])
