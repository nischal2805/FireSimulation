## Task 1: P2-001 Radiative heat transfer (modes 0/1/2) and fire spread between objects

**Files:** `src/livefire/sim/radiation.py` (new: Warp kernels + `RadiationModel`), `src/livefire/sim/surface.py`
(add `q_rad` input to `thermal_kernel`, call radiation once per step, energy deduction P2R3),
`src/livefire/sim/coupled.py` (`stats()` gains radiation counters), `configs/sim.yaml` (new keys),
`tests/test_radiation.py` (new).

- `configs/sim.yaml` new keys (section comment `# --- radiation (P2-001, PRD 8.2, ADR-0006)`):
  `rad_mode {0, "-", [0, 2], artistic}`, `eps_f {0.7, "-", [0, 1], artistic}`,
  `T_rad_flame {1100, "K", [600, 2000], artistic}`, `rad_R {0.30, "m", [0.02, 1.5], artistic}`,
  `rad_r_min {0.01, "m", [0.001, 0.1], artistic}`, `rad_cell {0.05, "m", [0.01, 0.3], artistic}`,
  `rad_max_emitters {8192, "-", [16, 1.0e6], artistic}` (capacity of the compacted emitter / active-cell list;
  overflow is counted, never silent).
- `RadiationModel(pos, normal, area, eps, two_sided, roi, params, device)` (host inputs float64; owns device arrays):
  `compute(burning: wp.array(uint8), T_s: wp.array(float64), out_q: wp.array(float64)) -> None` fills `out_q` (N,)
  [W/m^2 absorbed] per P2R2 for the configured mode; mode 0 writes zeros without launching gather kernels.
  Mode 2: emitter compaction with `wp.utils.array_scan` (P2R5), gather kernel over the compacted list with the
  `rad_R` cut-off. Mode 1: per-cell power, power-weighted centroid and normal accumulated on a grid of `rad_cell`
  (P2R4), active cells compacted the same way, gather kernel over active cells. Expose `last_emitters`,
  `overflow_count`, `last_absorbed_W` (sum `A_i q_rad_i`, syncs) for stats.
- `SurfaceFireModel`: constructs a `RadiationModel` in `set_material_ids` (two-sided flag = `materials.consumed`);
  public `compute_radiation() -> wp.array` (fills and returns the model-owned `q_rad` array; used by `step` and by the
  Task 2 bench). `thermal_kernel` gains a `q_rad: wp.array(dtype=wp.float64)` input added to `q`. After the
  combustion kernel, apply P2R3: scale `heat_W` of burning emitters down by `P_rx * w_b / sum(w)` (one small kernel;
  remainder into a float64 `rad_deficit_J` device counter). `clamp_events()` unchanged; new `radiation_stats()` ->
  `{"rad_emitters", "rad_overflow", "rad_absorbed_W", "rad_deficit_J"}`. Stability: `q_rad` is an explicit source
  term independent of `T_i`; `substeps_for` unchanged.
- `FireSimulation.stats()` merges `self.surface.radiation_stats()`; `stages()` unchanged (P2R6).
- Tests (`tests/test_radiation.py`, all fast; use the `grid_graph`/`sim_params` pattern from
  `tests/test_surface_invariants.py`, copying the two helpers into the new test file; `tests/` is not a package):
  1. Analytic view factor, mode 2: one emitter element (forced burning, `T_s = T_rad_flame`, area A) and one receiver
     at distance r = 0.10 m on the emitter's normal axis, facing it: `q_rad` equals
     `eps_i * eps_f * sigma * T^4 * A / (pi r^2)` within 1e-6 relative; rotating the receiver by 60 deg halves it
     (cos) within 1e-6; at r < `rad_r_min` the value saturates at `r_min`; beyond `rad_R` it is 0.
  2. One-sided vs two-sided receiver: a `wood` receiver facing away gets 0, a `paper` receiver facing away gets the
     full value.
  3. Mode 1 vs mode 2 on a 10x10 burning sheet and a parallel receiver sheet 0.15 m away: total absorbed power within
     15 % of each other; mode 0 gives exactly 0.
  4. Energy (P2R3): with gas temperature fixed at ambient, after one step `sum(heat_W) * dt + sum(A_i q_rad_i) * dt`
     equals the no-radiation `sum(heat_W) * dt` within 1e-6 relative when no clamp binds; `rad_deficit_J == 0`.
  5. Spread on `table_v1` (surface model only, `T_g` held at `T_amb`, dt 0.05 s, 30 s simulated, `ignite` the whole
     paper: point `(-0.075, 0.0, 0.1)`, r 0.15; with `T_g` frozen there is no gas-driven spread up the sheet, so a
     small ignition is not a fair radiation test): with `rad_mode=2` the cloth (obj 4) has burning
     elements before 30 s and the metal box (obj 2) never burns and keeps `m_fuel == 0`; with `rad_mode=0` the cloth
     never ignites in the same 30 s. Same assertions for `rad_mode=1` (cloth ignites). Print the cloth ignition times.
  6. Determinism: two identical mode-2 runs (40 steps) give identical `to_host()` arrays; the coupled CPU replay digest
     of two `FireSimulation` runs at G=32 with `rad_mode=2` (6 steps) is equal.
  7. Overflow: with `rad_max_emitters=16` and > 16 burning elements, `rad_overflow > 0` and no crash.
- Feasibility (controller estimate at plan time, P2R2 formula with all 280 paper elements burning, `T_b` = 1100 K,
  cloth near `T_ign` losing ~5.7 kW/m^2 convection + ~3.4 kW/m^2 re-radiation): peak absorbed flux on the cloth is
  ~16 kW/m^2 at `eps_f` 0.5 and ~22 kW/m^2 at 0.7; a lumped estimate ignites the nearest cloth after ~5.6 s (0.5) and
  ~3.7 s (0.7), against a paper burn time of `m0_areal / mdot_ref` ~ 6.7 s. Hence the default `eps_f` 0.7 (margin).
  This task may tune `eps_f`, `T_rad_flame` and `rad_R` within their ranges to make test 5 pass; if the cloth cannot
  ignite within the declared ranges, stop and report with the measured fluxes (do not change materials or widen
  ranges).
- Existing fast tests unchanged and green (default `rad_mode=0`).

**Acceptance:** `uv run pytest -m "not slow" -q` green; test 5 shows paper -> cloth spread with radiation and none
without; report lists the cloth ignition times per mode (CPU, from the test print).

