## Task 7: P0-007 Surface fire model (thermal, ignition, combustion, char)

**Files:** `src/livefire/sim/surface.py` (kernels + `SurfaceFireModel`), `src/livefire/sim/device.py` (device
selection and snapshot upload helpers), `configs/sim.yaml` (surface section), `tests/test_surface_invariants.py`.
Framework: **NVIDIA Warp** (`warp-lang`, ADR-0001, status proposed pending the GPU run). Warp's CPU backend is
single-threaded: keep test grids/element counts small (G <= 32 in unit tests) so `make test` stays fast.

- `SurfaceFireModel(snapshot, materials: MaterialTable, params, device)`: owns device arrays T_s, E, m_fuel, char,
  removed, burning, per-element material id (overridable), precomputed edge conductances `G (N,K)` (controller
  ruling formula), `Cap_i = A_i * C''_i`.
- `step(dt, T_g_sample (N,) device array) -> (heat_W (N,), soot_kg_s (N,))`: per element, in this order:
  1. thermal: `q_conv = h_c (T_g - T_s)`; conduction edge fluxes; `q_rr = -eps sigma (T_s^4 - T_amb^4)`;
     `q_fb = f_feedback * chi dH mdot''` while burning; `T_s += dt * (A (q_conv + q_rr + q_fb) + P_cond) / Cap`;
     clamp to `[T_amb_min, T_s_max]`, count clamps.
     Internal sub-stepping when `dt > dt_cond_max` (ruling) or when `dt * A h_c / Cap > 0.5` (convective stability);
     the number of sub-steps is exposed.
  2. ignition exposure per PRD 8.3 (`E += dt` if `T_s >= T_ign` else `E = max(0, E - decay dt)`); ignite when
     `E >= t_delay` and `m_fuel > m_res + eps_m` and flammable.
  3. combustion per PRD 8.3: `m_new = max(m_res, m_fuel - mdot_ref dt)`, `consumed = m_fuel - m_new`;
     `heat_W = (1 - f_feedback) * chi * dH * consumed * A / dt`; `soot_kg_s = y_soot * consumed * A / dt`;
     char per PRD formula (clamped 0..1; for `char_yield = 0` use `(m0 - m)/m0`); burnt out when
     `m_fuel <= m_res + eps_m` -> burning = 0; `removed = char if consumed_when_burnt else 0`.
  4. Non-flammables: never ignite, `m_fuel = 0` always, char/removed stay 0.
- `ignite(point, radius)`: elements within radius (flammable only) get `T_s = T_pilot`, `E = t_delay`; returns count.
- `sample_points()` returns `x_i + n_i * h_gas` (gas sampling/injection points, ruling).
- `configs/sim.yaml` surface params: `T_amb 293.15 K [250, 320] physical_estimate`; `h_c 25 W/(m^2 K) [1, 200]
  physical_estimate`; `f_feedback 0.3 - [0, 1] artistic`; `T_pilot 900 K [400, 1500] artistic`;
  `ignite_radius 0.03 m [0.005, 0.2] artistic`; `ignition_decay 0.5 1/s [0, 10] artistic`;
  `T_s_max 1500 K [800, 2500] artistic`; `T_s_min 250 K [200, 300] artistic`; `eps_m 1e-6 kg/m^2 [0, 1e-3] artistic`.
- Invariant tests (PRD 8.6), with T_g given as a constant or a prescribed field (no gas solver yet), on the Task 6
  snapshot and on tiny hand-built element graphs:
  fuel bounds `m_res <= m_fuel <= m0` and char/removed in [0,1] over a 120 s burn after ignition of paper;
  no spontaneous fire (T_g = T_amb, 600 s, nothing burns, T_s stays T_amb within 1e-3 K);
  non-flammables (metal, glass, ceramic, other) never ignite even at `T_g = 1500 K` for 60 s and after `ignite()`;
  conduction energy conservation on an isolated graph (h_c = 0, eps = 0 via test params, no burning): total
  `sum Cap_i T_i` conserved to 1e-6 relative over 1000 steps, and the temperature variance decreases monotonically;
  stability: large dt triggers sub-stepping and stays bounded;
  monotonicity smoke test: same geometry, paper vs cloth vs wood under the same T_g ramp ignites in order
  paper < cloth < wood (reports times; this is the M9 seed, not the P2 validation);
  determinism: two runs bitwise equal on CPU.

**Acceptance:** all invariant tests pass.

