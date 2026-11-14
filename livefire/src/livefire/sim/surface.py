"""Surface fire model (PRD 8.2, 8.3): thermal update, ignition, combustion, char. NVIDIA Warp kernels.

State and parameters are float64 on the device (conservation to 1e-6 needs it); the gas-facing
interface (``T_g_sample`` in, ``heat_W`` / ``soot_kg_s`` out) is float32, matching the grid fields.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from livefire.common.config import Params
from livefire.scene.snapshot import SceneSnapshot
from livefire.sim.device import select_device, to_device
from livefire.sim.materials import MaterialTable
from livefire.sim.state import FireStateHost

SIGMA = 5.670374419e-8  # W/(m^2 K^4)
_F64 = wp.float64


@wp.kernel
def thermal_kernel(
    t_in: wp.array(dtype=wp.float64),
    t_out: wp.array(dtype=wp.float64),
    t_gas: wp.array(dtype=wp.float32),
    nb: wp.array2d(dtype=wp.int32),
    gij: wp.array2d(dtype=wp.float64),
    cap: wp.array(dtype=wp.float64),
    area: wp.array(dtype=wp.float64),
    eps: wp.array(dtype=wp.float64),
    q_fb: wp.array(dtype=wp.float64),
    burning: wp.array(dtype=wp.uint8),
    clamps: wp.array(dtype=wp.int32),
    h_c: wp.float64,
    t_amb: wp.float64,
    t_min: wp.float64,
    t_max: wp.float64,
    dt: wp.float64,
    kmax: int,
):
    i = wp.tid()
    ti = t_in[i]
    sig = _F64(SIGMA)
    t4 = ti * ti * ti * ti
    ta4 = t_amb * t_amb * t_amb * t_amb
    q = h_c * (_F64(t_gas[i]) - ti) - eps[i] * sig * (t4 - ta4)
    if burning[i] != wp.uint8(0):
        q += q_fb[i]  # full q_fb even on the burnout step (partial consumption): small, accepted overestimate
    p = _F64(0.0)
    for s in range(kmax):
        j = nb[i, s]
        if j >= 0:
            p += gij[i, s] * (t_in[j] - ti)
    tn = ti + dt * (area[i] * q + p) / cap[i]
    if tn < t_min:
        tn = t_min
        wp.atomic_add(clamps, 0, 1)
    elif tn > t_max:
        tn = t_max
        wp.atomic_add(clamps, 0, 1)
    t_out[i] = tn


@wp.kernel
def combustion_kernel(
    t_s: wp.array(dtype=wp.float64),
    expo: wp.array(dtype=wp.float64),
    m_fuel: wp.array(dtype=wp.float64),
    char_f: wp.array(dtype=wp.float64),
    removed: wp.array(dtype=wp.float64),
    burning: wp.array(dtype=wp.uint8),
    area: wp.array(dtype=wp.float64),
    flam: wp.array(dtype=wp.uint8),
    consumed_flag: wp.array(dtype=wp.uint8),
    m0: wp.array(dtype=wp.float64),
    m_res: wp.array(dtype=wp.float64),
    t_ign: wp.array(dtype=wp.float64),
    t_delay: wp.array(dtype=wp.float64),
    mdot_ref: wp.array(dtype=wp.float64),
    dh: wp.array(dtype=wp.float64),
    chi: wp.array(dtype=wp.float64),
    y_soot: wp.array(dtype=wp.float64),
    heat_w: wp.array(dtype=wp.float32),
    soot: wp.array(dtype=wp.float32),
    decay: wp.float64,
    f_fb: wp.float64,
    eps_m: wp.float64,
    dt: wp.float64,
):
    i = wp.tid()
    heat_w[i] = 0.0
    soot[i] = 0.0
    if flam[i] == wp.uint8(0):
        return
    # 2. ignition exposure (PRD 8.3)
    e = expo[i]
    if t_s[i] >= t_ign[i]:
        e += dt
    else:
        e = wp.max(_F64(0.0), e - decay * dt)
    expo[i] = e
    mf = m_fuel[i]
    if burning[i] == wp.uint8(0) and e >= t_delay[i] and mf > m_res[i] + eps_m:
        burning[i] = wp.uint8(1)
    # 3. combustion
    if burning[i] != wp.uint8(0):
        mn = wp.max(m_res[i], mf - mdot_ref[i] * dt)
        if mn <= m_res[i] + eps_m:
            mn = m_res[i]  # burnt out: account the remainder, stop burning
            burning[i] = wp.uint8(0)
        used = mf - mn
        m_fuel[i] = mn
        heat_w[i] = wp.float32((_F64(1.0) - f_fb) * chi[i] * dh[i] * used * area[i] / dt)
        soot[i] = wp.float32(y_soot[i] * used * area[i] / dt)
        den = m0[i] - m_res[i]
        c = _F64(0.0)
        if den > _F64(0.0):
            c = wp.clamp((m0[i] - mn) / den, _F64(0.0), _F64(1.0))
        char_f[i] = c
        if consumed_flag[i] != wp.uint8(0):
            removed[i] = c


class SurfaceFireModel:
    """Surface thermal/ignition/combustion/char model for N_e surface elements.

    Owns device arrays (float64, shape (N,)): ``T_s`` [K], ``E`` ignition exposure [s], ``m_fuel``
    [kg/m^2], ``char`` and ``removed`` [0..1], ``burning`` (uint8), plus per-element material ids
    (overridable via :meth:`set_material_ids`), edge conductances ``G`` (N,K) [W/K] and
    ``Cap`` = A * C'' [J/K]. Gas coupling happens at :meth:`sample_points` (world frame, m).
    Inputs/outputs of :meth:`step` are float32 (N,) arrays on the model's device.
    """

    def __init__(
        self, snapshot: SceneSnapshot, materials: MaterialTable, params: Params, device: str | None = None
    ) -> None:
        self.device = select_device(device)
        self.params = params
        self.materials = materials
        self.n = int(snapshot.pos.shape[0])
        self.pos = np.asarray(snapshot.pos, dtype=np.float64)
        self.normal = np.asarray(snapshot.normal, dtype=np.float64)
        self.h_gas = float(snapshot.h_gas)
        self._area_h = np.asarray(snapshot.area, dtype=np.float64)
        self._nb_h = np.asarray(snapshot.neighbors, dtype=np.int32)
        self._spacing = float(snapshot.surfel_spacing)
        self._p = {k: params[k] for k in params.keys()}  # noqa: SIM118 (Params is not iterable)
        d = self.device
        self.area = to_device(self._area_h, wp.float64, d)
        self.nb = to_device(self._nb_h, wp.int32, d)
        self.clamp_count = wp.zeros(1, dtype=wp.int32, device=d)
        self.heat_W = wp.zeros(self.n, dtype=wp.float32, device=d)
        self.soot_kg_s = wp.zeros(self.n, dtype=wp.float32, device=d)
        self.t_sim = 0.0
        self.step_count = 0
        self.last_substeps = 0
        self.max_substeps = 0
        self.set_material_ids(np.asarray(snapshot.material_id, dtype=np.int64))

    # ---- setup ----------------------------------------------------------------------------
    def set_material_ids(self, ids: np.ndarray) -> None:
        """(Re)assign per-element material ids (N,) and reset the fire state: T_s = T_amb, no
        exposure, m_fuel = m0, char = removed = 0, nothing burning."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != (self.n,):
            raise ValueError(f"material ids must have shape ({self.n},)")
        m, d, p = self.materials, self.device, self._p
        self.material_id = ids.copy()
        f64 = lambda a: to_device(np.asarray(a, dtype=np.float64)[ids], wp.float64, d)
        u8 = lambda a: to_device(np.asarray(a, dtype=np.uint8)[ids], wp.uint8, d)
        self._flam_h = np.asarray(m.flammable)[ids]
        self.flam, self.consumed_flag = u8(m.flammable), u8(m.consumed)
        self.eps = f64(m.eps)
        self.m0, self.m_res = f64(m.m0), f64(m.m_res)
        self.t_ign, self.t_delay, self.mdot_ref = f64(m.T_ign), f64(m.t_delay), f64(m.mdot_ref)
        self.dh, self.chi, self.y_soot = f64(m.dH), f64(m.chi), f64(m.y_soot)
        self.q_fb = f64(p["f_feedback"] * m.chi * m.dH * m.mdot_ref)
        self._t_delay_h = np.asarray(m.t_delay)[ids]
        self._m0_h = np.asarray(m.m0)[ids]
        self._m_res_h = np.asarray(m.m_res)[ids]
        # edge conductances G_ij = kd_ij * s / d_ij, kd harmonic mean of k*delta (controller ruling)
        kd = (np.asarray(m.k) * np.asarray(m.delta))[ids]
        nb = self._nb_h
        j = np.where(nb >= 0, nb, 0)
        valid = nb >= 0
        ki, kj = kd[:, None], kd[j]
        kdij = 2.0 * ki * kj / (ki + kj)
        dist = np.linalg.norm(self.pos[:, None, :] - self.pos[j], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(valid, kdij * self._spacing / dist, 0.0)
        self._g_h = g
        cap = self._area_h * np.asarray(m.C_areal)[ids]
        self._cap_h = cap
        self.G = to_device(g, wp.float64, d)
        self.Cap = to_device(cap, wp.float64, d)
        # stability limits
        rate_cond = g.sum(axis=1) / cap
        self.dt_cond_max = 0.5 / rate_cond.max() if rate_cond.size and rate_cond.max() > 0 else math.inf
        cpp = np.asarray(m.C_areal)[ids]
        h_eff = p["h_c"] + 4.0 * np.asarray(m.eps)[ids] * SIGMA * p["T_s_max"] ** 3  # conv + linearised re-radiation
        r_conv = (h_eff / cpp).max() if self.n else 0.0
        self.dt_conv_max = 0.5 / r_conv if r_conv > 0 else math.inf
        self.reset()

    def reset(self) -> None:
        """Reset the dynamic state (see :meth:`set_material_ids`) and counters."""
        d, p = self.device, self._p
        self.T_s = to_device(np.full(self.n, p["T_amb"]), wp.float64, d)
        self._T_tmp = wp.zeros(self.n, dtype=wp.float64, device=d)
        self.E = wp.zeros(self.n, dtype=wp.float64, device=d)
        self.m_fuel = to_device(self._m0_h, wp.float64, d)
        self.char = wp.zeros(self.n, dtype=wp.float64, device=d)
        self.removed = wp.zeros(self.n, dtype=wp.float64, device=d)
        self.burning = wp.zeros(self.n, dtype=wp.uint8, device=d)
        self.clamp_count.zero_()
        self.t_sim, self.step_count, self.last_substeps, self.max_substeps = 0.0, 0, 0, 0

    # ---- interface ------------------------------------------------------------------------
    def sample_points(self) -> np.ndarray:
        """Gas sampling/injection points ``x_i + n_i * h_gas``, (N,3) float32, world frame [m]."""
        return (self.pos + self.normal * self.h_gas).astype(np.float32)

    def ignite(self, point: np.ndarray, radius: float | None = None) -> int:
        """User ignition: flammable elements within ``radius`` [m] (default ``ignite_radius``) of world
        ``point`` get ``T_s = T_pilot``, ``E = t_delay`` and ``burning = 1``. Returns the number of elements affected."""
        r = self._p["ignite_radius"] if radius is None else float(radius)
        dist = np.linalg.norm(self.pos - np.asarray(point, dtype=np.float64), axis=1)
        hit = (dist <= r) & self._flam_h & (self.m_fuel.numpy() > self._m_res_h + self._p["eps_m"])
        if hit.any():
            t, e, b = self.T_s.numpy().copy(), self.E.numpy().copy(), self.burning.numpy().copy()
            t[hit] = self._p["T_pilot"]
            e[hit] = self._t_delay_h[hit]
            b[hit] = 1  # R7: lit elements burn immediately, robust to the next thermal step
            self.T_s.assign(t)
            self.E.assign(e)
            self.burning.assign(b)
        return int(hit.sum())

    def substeps_for(self, dt: float) -> int:
        """Number of thermal sub-steps used for a step of ``dt`` seconds."""
        return max(1, math.ceil(dt / min(self.dt_cond_max, self.dt_conv_max) - 1e-9))

    def step(self, dt: float, T_g_sample: wp.array) -> tuple[wp.array, wp.array]:
        """Advance ``dt`` [s]. ``T_g_sample`` (N,) float32 [K], on the model device, is the gas
        temperature at :meth:`sample_points`. Returns ``(heat_W, soot_kg_s)``: float32 (N,) device
        arrays [W], [kg/s] owned by the model and overwritten by the next call."""
        p, d = self._p, self.device
        n_sub = self.substeps_for(dt)
        h = dt / n_sub
        for _ in range(n_sub):
            wp.launch(
                thermal_kernel, dim=self.n, device=d,
                inputs=[self.T_s, self._T_tmp, T_g_sample, self.nb, self.G, self.Cap, self.area, self.eps,
                        self.q_fb, self.burning, self.clamp_count, p["h_c"], p["T_amb"], p["T_s_min"],
                        p["T_s_max"], h, self.nb.shape[1]],
            )  # fmt: skip
            self.T_s, self._T_tmp = self._T_tmp, self.T_s
        wp.launch(
            combustion_kernel, dim=self.n, device=d,
            inputs=[self.T_s, self.E, self.m_fuel, self.char, self.removed, self.burning, self.area,
                    self.flam, self.consumed_flag, self.m0, self.m_res, self.t_ign, self.t_delay,
                    self.mdot_ref, self.dh, self.chi, self.y_soot, self.heat_W, self.soot_kg_s,
                    p["ignition_decay"], p["f_feedback"], p["eps_m"], dt],
        )  # fmt: skip
        self.last_substeps = n_sub
        self.max_substeps = max(self.max_substeps, n_sub)
        self.t_sim += dt
        self.step_count += 1
        return self.heat_W, self.soot_kg_s

    # ---- readback -------------------------------------------------------------------------
    def clamp_events(self) -> int:
        """Cumulative number of T_s clamp events since :meth:`reset` (syncs)."""
        return int(self.clamp_count.numpy()[0])

    def to_host(self) -> dict[str, np.ndarray]:
        """Copy of the dynamic state as NumPy float64 arrays (syncs); caller owns the copies."""
        wp.synchronize_device(self.device)
        return {
            "T_s": self.T_s.numpy().copy(), "E": self.E.numpy().copy(), "m_fuel": self.m_fuel.numpy().copy(),
            "char": self.char.numpy().copy(), "removed": self.removed.numpy().copy(),
            "burning": self.burning.numpy().copy(),
        }  # fmt: skip

    def fire_state_host(self, T_g: np.ndarray, vel: np.ndarray, soot: np.ndarray, p: np.ndarray) -> FireStateHost:
        """Assemble a ``FireStateHost`` (float32 copies) from this model plus caller-supplied gas fields."""
        s = self.to_host()
        f = lambda a: a.astype(np.float32)
        return FireStateHost(
            T_s=f(s["T_s"]), ignition_exposure=f(s["E"]), m_fuel=f(s["m_fuel"]), char=f(s["char"]),
            removed=f(s["removed"]), burning=s["burning"].astype(np.uint8), T_g=T_g, vel=vel, soot=soot,
            p=p, t_sim=self.t_sim, step=self.step_count,
        )  # fmt: skip
