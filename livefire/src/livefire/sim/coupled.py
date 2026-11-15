"""Coupled fire simulation: surface fire model + gas solver on one device (PRD 8, 5.3).

``FireSimulation`` implements ``SimulationBackend`` (reset/ignite/step/state_host/set_material) and
``FireStateView`` (device fields for the renderer). Fixed ``dt_sim``; ``run_frame`` consumes wall time through
an accumulator and never changes ``dt``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import warp as wp

from livefire.common.config import Params
from livefire.common.geometry import AABB
from livefire.scene.snapshot import SceneSnapshot
from livefire.sim.device import select_device, to_device
from livefire.sim.gas import GasSolver
from livefire.sim.materials import MaterialTable
from livefire.sim.state import FireStateHost
from livefire.sim.surface import SurfaceFireModel

_RT_WINDOW = 60  # frames in the rt_factor sliding window


@wp.kernel
def _cast_f32_k(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float32(a[i])


class FrameResult(NamedTuple):
    """``run_frame`` result: steps taken, ``rt_factor`` (simulated s / wall s, sliding window), ``sim_lag_s``
    (accumulated wall time not yet simulated, s)."""

    steps: int
    rt_factor: float
    sim_lag_s: float


class FireSimulation:
    """Surface fire model coupled to the MAC gas solver.

    Per step: sample ``T_g`` at the element sample points (``x + n h_gas``), ``surface.step``, inject
    ``heat_J = heat_W dt`` and ``soot_kg = soot_kg_s dt`` at the same points, ``gas.step``. Device = ``app_cfg
    ["device"]`` (``cpu`` here). Not thread-safe (single owner). The snapshot is never mutated.

    ``FireStateView`` fields (device arrays, valid until the next step, read-only for the renderer): ``T_g`` (G,G,G)
    float32 K, ``soot`` (G,G,G) float32 kg/m^3 (mass concentration), ``flame`` (G,G,G) float32 W/m^3 (combustion
    intensity, ADR-0004), ``char`` / ``removed`` (N_e,) float32 0..1
    (fresh float32 copies of the float64 model state made on access), ``h_gas`` m, ``roi`` AABB.
    """

    def __init__(self, snapshot: SceneSnapshot, materials: MaterialTable, sim_params: Params, app_cfg: dict[str, Any]):
        self.materials = materials
        self.params = sim_params
        self.device = select_device(str(app_cfg.get("device", "cpu")))
        self.dt_sim = float(app_cfg["dt_sim"])
        self.max_substeps = int(app_cfg["max_substeps_per_frame"])
        self.seed = int(app_cfg.get("seed", 0))
        self.latency_ms = 0.0
        self._ready = False
        self.reset(snapshot)

    # ---- SimulationBackend ------------------------------------------------------------------
    def reset(self, snapshot: SceneSnapshot) -> None:
        """Upload ``snapshot`` to device mirrors owned by this object and reset all state."""
        self.snapshot = snapshot
        G = int(snapshot.solid_mask.shape[0])
        self.gas = GasSolver(G, snapshot.h_gas, snapshot.roi.min, snapshot.solid_mask, self.params, self.device, self.seed)
        self.surface = SurfaceFireModel(snapshot, self.materials, self.params, self.device)
        self._pts = to_device(self.surface.sample_points(), wp.float32, self.device)
        self._t_samp = wp.zeros(self.surface.n, dtype=wp.float32, device=self.device)
        self._char32 = wp.zeros(self.surface.n, dtype=wp.float32, device=self.device)
        self._removed32 = wp.zeros(self.surface.n, dtype=wp.float32, device=self.device)
        self._step_count = 0
        self._acc = 0.0
        self._window: deque[tuple[float, float]] = deque(maxlen=_RT_WINDOW)
        self._ready = True

    def _check(self) -> None:
        if not self._ready:
            raise RuntimeError("FireSimulation.reset(snapshot) has not been called")

    @property
    def t_sim(self) -> float:
        """Simulated time [s] = ``step_count * dt_sim``."""
        return self._step_count * self.dt_sim

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def peak_vram_allocated_mb(self) -> float:
        return 0.0 if self.device == "cpu" else wp.get_mempool_used_mem_high(self.device) / 1e6

    @property
    def peak_vram_reserved_mb(self) -> float:
        return self.peak_vram_allocated_mb

    def ignite(self, point_world: np.ndarray, radius_m: float) -> int:
        """Ignite flammable elements within ``radius_m`` [m] of world ``point_world`` (3,); returns the number lit."""
        self._check()
        p = np.asarray(point_world, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all() or not radius_m > 0:
            raise ValueError("ignite needs a finite (3,) point and radius > 0")
        return self.surface.ignite(p, float(radius_m))

    def stages(self, dt: float) -> dict[str, Callable[[], None]]:
        """Ordered per-stage callables of one coupled step of ``dt`` [s]: ``sample_gas`` (gas T at the element
        sample points), ``surface`` (thermal + combustion, keeps the sources), ``gas`` (inject sources + gas step).
        Calling them in order is exactly one :meth:`step` iteration (``step`` runs these same callables) except for
        the step counter, which only ``step`` advances. No device sync inside; the caller syncs."""
        s, g = self.surface, self.gas
        src: list[wp.array] = []

        def sample_gas() -> None:
            g.sample_T(self._pts, self._t_samp)

        def surface() -> None:
            src[:] = s.step(dt, self._t_samp)

        def gas() -> None:
            if not src:
                raise RuntimeError("stage 'gas' needs a preceding 'surface' stage")
            g.step(dt, self._pts, src[0], src[1], s.released_power_W)

        return {"sample_gas": sample_gas, "surface": surface, "gas": gas}

    def step(self, n_steps: int) -> None:
        """Advance exactly ``n_steps`` of ``dt_sim``; returns when the device work is complete."""
        self._check()
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        t0 = time.perf_counter_ns()
        stages = list(self.stages(self.dt_sim).values())
        for _ in range(n_steps):
            for stage in stages:
                stage()
            self._step_count += 1
        wp.synchronize_device(self.device)
        if n_steps:
            self.latency_ms = (time.perf_counter_ns() - t0) / 1e6 / n_steps

    def run_frame(self, wall_dt: float) -> FrameResult:
        """Accumulate ``wall_dt`` [s] and advance ``min(max_substeps_per_frame, accumulated / dt_sim)`` fixed
        steps. ``rt_factor`` is simulated s / wall s over the last 60 frames."""
        self._check()
        if not wall_dt >= 0:
            raise ValueError("wall_dt must be >= 0")
        self._acc += wall_dt
        n = min(self.max_substeps, int(self._acc / self.dt_sim + 1e-9))
        self.step(n)
        self._acc -= n * self.dt_sim
        self._window.append((n * self.dt_sim, wall_dt))
        sim_s = sum(a for a, _ in self._window)
        wall_s = sum(b for _, b in self._window)
        return FrameResult(n, sim_s / wall_s if wall_s > 0 else 0.0, self._acc)

    def set_material(self, obj_id: int, material_id: int) -> None:
        """Re-assign all elements of world object ``obj_id`` to ``material_id`` (0..7). Those elements restart
        fresh (fuel ``m0`` of the new material, not burning, ``T_s`` kept); other elements keep their state."""
        self._check()
        if not 0 <= material_id < len(self.materials.names):
            raise ValueError(f"material_id {material_id} out of range")
        mask = self.snapshot.obj_id == obj_id
        if not mask.any():
            raise ValueError(f"no elements with obj_id {obj_id}")
        old = self.surface.to_host()
        ids = self.surface.material_id.copy()
        ids[mask] = material_id
        self.surface.set_material_ids(ids)  # resets all state; restore the untouched elements below
        keep = ~mask
        new = self.surface.to_host()
        new["T_s"] = old["T_s"]
        for k in ("E", "m_fuel", "char", "removed", "burning"):
            new[k][keep] = old[k][keep]
        for name, dev in (("T_s", self.surface.T_s), ("E", self.surface.E), ("m_fuel", self.surface.m_fuel),
                          ("char", self.surface.char), ("removed", self.surface.removed),
                          ("burning", self.surface.burning)):  # fmt: skip
            dev.assign(new[name])

    def state_host(self) -> FireStateHost:
        """Host copy of the full state. ``vel`` is cell-centred (G,G,G,3) m/s (ADR-0002); ``p`` in Pa."""
        self._check()
        wp.synchronize_device(self.device)
        g = self.gas
        st = self.surface.fire_state_host(
            g.T_g.numpy().copy(), g.velocity_cell_centered(), g.soot.numpy().copy(),
            (g.p.numpy() * self.params["rho_g"]).astype(np.float32), g.flame.numpy().copy(),
        )  # fmt: skip
        st.t_sim, st.step = self.t_sim, self._step_count
        return st

    def stats(self) -> dict[str, float]:
        """Clamp counters, last-step divergence norms before/after projection, sub-steps, timing (syncs)."""
        self._check()
        c = self.gas.counters()
        b, a = self.gas.divergence_norms()
        wall = sum(w for _, w in self._window)
        return {
            **c,
            "T_s_clamp": self.surface.clamp_events(),
            "div_before": b,
            "div_after": a,
            "surface_substeps": self.surface.last_substeps,
            "surface_max_substeps": self.surface.max_substeps,
            "step_count": self._step_count,
            "t_sim": self.t_sim,
            "rt_factor": sum(s for s, _ in self._window) / wall if wall > 0 else 0.0,
            "sim_lag_s": self._acc,
            **self.surface.radiation_stats(),
        }

    # ---- FireStateView ----------------------------------------------------------------------
    @property
    def T_g(self) -> wp.array:
        return self.gas.T_g

    @property
    def soot(self) -> wp.array:
        return self.gas.soot

    @property
    def flame(self) -> wp.array:
        return self.gas.flame

    @property
    def char(self) -> wp.array:
        wp.launch(_cast_f32_k, dim=self.surface.n, inputs=[self.surface.char, self._char32], device=self.device)
        return self._char32

    @property
    def removed(self) -> wp.array:
        wp.launch(_cast_f32_k, dim=self.surface.n, inputs=[self.surface.removed, self._removed32], device=self.device)
        return self._removed32

    @property
    def h_gas(self) -> float:
        return float(self.snapshot.h_gas)

    @property
    def roi(self) -> AABB:
        return self.snapshot.roi
