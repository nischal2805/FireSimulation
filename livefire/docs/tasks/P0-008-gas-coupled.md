## Task 8: P0-008 Gas solver baseline and coupled simulation

**Files:** `src/livefire/sim/gas.py` (`GasSolver`, MAC grid), `src/livefire/sim/coupled.py` (`FireSimulation`
implementing `SimulationBackend`), `configs/sim.yaml` (gas section), `tests/test_gas.py`, `tests/test_coupled.py`,
`docs/decisions/ADR-0002-mac-grid.md`, `docs/decisions/ADR-0003-energy-split.md`.

- MAC grid G^3, `h = roi_size / G`: faces u (G+1,G,G), v (G,G+1,G), w (G,G,G+1); cell-centred T_g, soot, p,
  solid (from snapshot). Boundaries: domain walls and solid cells are no-through (face velocity 0 on faces touching
  a solid cell or a wall), except the top face (z = roi_max) which is open: pressure Dirichlet p = 0 and outflow
  allowed.
- Per step (`dt`): (1) advect u, v, w, T_g, soot semi-Lagrangian, RK2 backtrace, trilinear, clamp-to-domain;
  (2) add sources: `T_g += heat_J / (rho_g c_p h^3)` and `soot += soot_kg / h^3` scattered trilinearly at the
  element sample points (deterministic on CPU; on GPU atomics allowed, replay tolerance stated);
  (3) buoyancy on w faces `a_z = g (T_face - T_amb) / T_amb`, `g = 9.81`; (4) vorticity confinement strength
  `eps_vc`; (5) wind: inflow velocity on the -x wall plus seeded low-amplitude noise, default off (zero);
  (6) velocity clamp `|u| <= cfl_max h / dt` with a violation counter; (7) pressure projection, red-black
  Gauss-Seidel with `n_iters` iterations, solids as Neumann boundaries; record `||div u||_2` over fluid cells before
  and after; (8) cooling `T_g = T_amb + (T_g - T_amb) exp(-k_cool dt)`, soot `*= exp(-k_soot dt)`, clamp
  `T_g <= T_g_max` with counter. Accessor `velocity_cell_centered() -> (G,G,G,3)`.
- `configs/sim.yaml` gas params: `rho_g 1.2 kg/m^3 physical_estimate`, `c_p 1005 J/(kg K) physical_estimate`,
  `g 9.81 m/s^2`, `eps_vc 0.3 - artistic [0, 2]`, `k_cool 1.5 1/s artistic [0, 20]`, `k_soot 0.3 1/s artistic [0, 10]`,
  `T_g_max 2000 K artistic [1000, 3000]`, `cfl_max 3.0 - artistic [0.5, 10]`, `n_iters 60 - artistic [5, 500]`,
  `wind_speed 0.0 m/s artistic [0, 5]`, `wind_noise 0.0 m/s artistic [0, 1]`. Ranges for those not given: choose
  sensible ones and list them in the report.
- `FireSimulation(snapshot, materials, sim_params, app_cfg)`: fixed `dt_sim`; per `step`: sample T_g at element sample
  points (trilinear), `surface.step`, inject sources, `gas.step`. `run_frame(wall_dt)` advances
  `min(max_substeps_per_frame, accumulated / dt_sim)` steps (accumulator; never variable dt) and returns
  `rt_factor` (simulated s / wall s over a sliding window) and `sim_lag_s`. `ignite`, `set_material`,
  `state_host()` (FireStateHost incl. cell-centred vel), `stats()` (clamp counters, divergence norms, substeps).
- Tests: projection reduces `||div||_2` by >= 10x on a seeded random field at G=32 with default `n_iters`
  (report the actual factor; if 10x needs more iterations, report and set the default accordingly); no flux through
  solid faces after projection (exactly 0); zero-wind symmetry: a hot blob at the domain's x-y centre, no solids,
  60 steps at G=32: centre of mass of `T_g - T_amb` drifts < 0.25 h in x and y, and the field is mirror-symmetric
  in x within 1e-3 relative L2; buoyancy: a hot blob rises (centre of mass z increases); long run (slow marker):
  coupled sim on a down-scaled table_v1 snapshot at G=32, paper ignited, 10 min simulated (36000 steps): no NaN/Inf,
  all fields bounded (T_g <= T_g_max, |u| bounded by the clamp, soot >= 0), invariants of Task 7 still hold;
  replay: two coupled runs with the same seed and events give equal `FireStateHost.digest()` on CPU;
  resolution sensitivity (R9, non-gating, `xfail(strict=False)` if it fails): ignition time of the cloth when the
  paper is lit, compared at G=32 vs G=64 in this container (G=64 vs G=128 on the GPU machine); report the numbers.
- ADR-0002 (MAC storage, accessor keeps the public `vel` shape) and ADR-0003 (energy split, re-radiation, clamps)
  in the Appendix B format, status accepted, with PRD change-log entries appended to `docs/PRD.md`.

**Acceptance:** all non-slow tests pass; the slow long-run test passes once (`make test-slow`, wall time reported).

