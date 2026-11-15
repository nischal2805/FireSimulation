## Task 2: P2-002 Radiation comparison, ADR-0006, default mode, thermal/unit audit

**Files:** `bench/p2_radiation_compare.py` (new), `docs/decisions/ADR-0006-radiation.md` (new),
`configs/sim.yaml` (default `rad_mode` and tuned artistic radiation values), `configs/materials.yaml`
(`status_detail`, header comments), `src/livefire/sim/materials.py` (validate `status_detail`),
`bench/workloads.yaml` (`workload_version: 2`, P2R7), `tests/test_thermal_audit.py` (new),
`tests/test_materials.py` (cases for `status_detail`), `docs/PRD.md` (change log v2.3 only), Makefile `bench-radiation`
target + README row.

- `bench/p2_radiation_compare.py [--quick]`: on `table_v1` (default grid G from `configs/grid.yaml`, surface model with
  `T_g` = ambient for the spread metric; coupled at G=32 for the end-to-end variant), for each `rad_mode` in 0, 1, 2
  and the same paper ignition: (a) cost: median/p95 of `SurfaceFireModel.compute_radiation()` and of a full
  `surface` step at 3 fire sizes (number of burning elements ~100, ~1000, ~3000, reached by igniting radius 0.03,
  0.08, 0.15 m), >= 5 repeats, warm-up discarded (PRD 11.4) -> `bench/results/radiation_cpu_<date>.csv` (Appendix C
  rows, `stage=radiation|surface_step`, `variant=mode<k>_burning<n>`); (b) effect: cloth and wood first-ignition time,
  burnt area per object at 10/30/60 s, metal/glass never burning -> `eval/results/radiation_effect_<date>.csv`.
  Record the wall time of the run. `--quick` (tests) uses 1 repeat, small fire.
- ADR-0006 (Appendix B template): context (HANDOVER risk 3), candidates = PRD 8.2 options 1-4 in that order (4 is
  deferred with the reason), method, results table quoting both CSVs, PRD 11.6-style scoring adapted to a technique
  choice (visual effect = accuracy row, CPU cost now + "GPU cost pending user run", robustness, implementation risk),
  decision = the simplest option that makes cloth ignite from burning paper within the effect CSV's numbers
  ("simplest sufficient wins", PRD 8.2), fallback = the runner-up mode via `rad_mode`, consequences (P2R3 energy
  accounting, workload_version 2, resolution invariant uses the chosen mode). Status **accepted** for the numerics,
  with GPU cost listed as a pending measurement.
- Set the chosen default `rad_mode` and any tuned artistic values in `configs/sim.yaml` (tuning only within the
  declared ranges; document every change in the ADR). Re-run the fast tests: P0 tests whose expectations depended
  on "no spread" (search `tests/` for cloth/obj 4 assertions) are updated with a one-line reason.
- Audit (PRD 8.1, P2R8): every `configs/sim.yaml` entry has a non-empty unit and a status; every material in
  `configs/materials.yaml` gets `status_detail`; units header kept; `MaterialTable` validates `status_detail` keys
  (`thermal|ignition|combustion|radiation`) and values, rejects unknown keys, accepts its absence.
  `docs/NUMERICAL_METHODS.md` gets a "Parameter audit" table (parameter group, unit, status, source note) - Task 6
  writes the rest of that doc; this task adds only this table.
- `tests/test_thermal_audit.py` (fast, unit checks of the thermal model against closed forms):
  1. Radiative lumped heating: one receiver element (no neighbours, `h_c = 0`, gas at ambient) facing a small
     burning emitter patch 0.05 m away (`f_feedback = 0` so the emitter's `T_s <= T_pilot < T_rad_flame` and its
     emission is constant). The receiver's `T_s(t)` matches a `scipy.integrate.solve_ivp` solution of
     `C'' dT/dt = q_abs - eps sigma (T^4 - T_amb^4)` (with `q_abs` read from `compute_radiation()`) within 1 % at
     5 time points, and its ignition time matches within 2 % + one step, for paper and wood.
  2. Convective relaxation: `h_c = 25`, no radiation, gas fixed at 400 K: `T_s(t)` follows
     `T_g + (T_amb - T_g) exp(-h_c t / C'')` within 1 % at 3 time points (re-radiation off via `eps = 0`).
  3. Units sanity: `MaterialTable.C_areal` equals `rho * c * delta` for every class; derived `dt_cond_max` for the
     `table_v1` snapshot is positive and `substeps_for(dt_sim)` is finite.
  4. `sim.yaml` audit test: every parameter's `unit` is a non-empty string and `status` valid (via `load_params`).
- PRD change log: add `v2.3 (P2-002): ADR-0006 radiation (rad_mode, energy deduction from the gas share, eps_i
  absorptivity, |cos| for thin sheets); configs/sim.yaml radiation keys; materials.yaml optional status_detail;
  workload_version 2.` Do not change any other PRD text.

**Acceptance:** fast tests green; both CSVs committed; ADR-0006 quotes them; default mode set; `make bench-radiation`
documented in the README table.

