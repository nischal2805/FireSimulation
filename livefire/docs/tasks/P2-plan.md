# P2 implementation plan (simulation quality: radiation and spread, invariants, M9)

Spec: `docs/PRD.md` (authority; P2 = section 12 "P2: simulation quality", 8.1-8.6, M9 in 11.1, ablations in 11.2).
Rules: `CLAUDE.md`. Context: `docs/HANDOVER.md` (risk 3: no spread between objects), `docs/NUMERICAL_METHODS.md`,
ADR-0002/0003/0004. **Gate 2:** all PRD 8.6 invariants pass; M9 passes; numerical behaviour documented.
User-visible value first: Task 1 makes fire spread between objects (paper -> cloth) through radiation.

## Global Constraints (bind every task)

- Everything in the P0 and P1 plans' Global Constraints still holds (`docs/tasks/P0-plan.md`, `docs/tasks/P1-plan.md`,
  top sections): Python 3.11 + uv, geometry contract (PRD 4) verbatim, every tunable in YAML under `configs/` with
  `{value, unit, range, status}` (`physical_estimate|artistic`), seeded randomness, mocks marked `# MOCK:`, Warp for
  kernels, device from `configs/app.yaml: device`, ruff clean (`uv run ruff check src tests bench eval`), docstrings with
  shapes/units/frame/device/ownership.
- **No GPU, no camera, no display in the dev container.** Warp runs on its single-threaded CPU device. Measured CPU
  costs (from `bench/results/baseline_cpu_2026-11-14.csv`): coupled step ~0.66 s at G=64, ~5.9 s at G=128; surface
  step ~0.6 ms with 8292 elements. Therefore: fast tests use the surface model alone with a prescribed gas
  temperature, or the coupled sim at G <= 32; anything over ~30 s wall gets `@pytest.mark.slow`. **Numerical results
  computed on CPU are valid results** (only timings are hardware-specific); CPU timings are labelled CPU
  (`gpu=none`, `framework=warp-<ver>-cpu`).
- Run fast tests with `uv run pytest -m "not slow" -q` from `livefire/`. **Never run the full suite / `make test-slow`.**
  Run a slow test only by node id (e.g. `uv run pytest -q tests/test_p2_resolution.py::test_x`), and run multi-minute
  jobs (eval scripts, slow tests) with the Bash tool's background mode, recording the wall time.
- User machine: Windows 11 + RTX 4060. pathlib only, no POSIX-only calls, no shell scripts. Every new command goes in
  both the Makefile and the README "Windows (no make)" table. Do **not** edit the Makefile `eval` stub (P1-007 owns it);
  add new targets.
- Do not touch P1-006/P1-007 files being written concurrently (`app/live.py`, `app/change_monitor.py`,
  `render/geometry_overlay.py`, `app/measure_cli.py`, `eval/{alignment,geometry_mae,placement,latency}.py`,
  `tests/test_live_pipeline.py`, `tests/test_eval_p1.py`, `docs/RUNBOOK_P1.md`). P2 does not depend on live mode.
- Output locations: benchmark CSVs `bench/results/<name>_cpu_<YYYY-MM-DD>.csv` (Appendix C schema via
  `livefire.common.bench_csv` where the row is a timing); eval CSVs `eval/results/<name>_<YYYY-MM-DD>.csv` (create the
  folder; committed). Eval scripts live in `eval/p2_*.py`, runnable as `uv run python eval/p2_<name>.py`.
- Per task: copy the brief to `docs/tasks/<P2-ID>-<slug>.md`; commit `<P2-ID>: <title>` with trailers
  `Co-Authored-By: <your model name> <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01FvsiHx5LCL7MYc9rf6PQvv`; `git add` only the task's own files
  (never `-A`); do not push; do not edit `docs/STATUS.md` (controller-owned).
- Numbers in reports and ADRs only from committed CSVs. Do not silently change PRD contracts (thesis, section 4,
  section 9 interfaces, section 11 metric definitions, section 8.6 invariants): ADR + PRD change-log entry instead.
  Only Task 2 edits the PRD change log (entry v2.3); any later contract change needs its own ADR and is raised to the
  controller first.
- Existing tests must stay green. If a P2 change legitimately changes an existing expectation (e.g. a stage list or
  a benchmark workload), say so in the report and update the test in the same commit with a one-line reason.

## Controller rulings (made at plan time)

- P2R1 **Radiation mode key.** `configs/sim.yaml` is validated by `load_params` (numeric values only), so the mode is
  a numeric code: `rad_mode: {value: 0, unit: "-", range: [0, 2], status: artistic}` with 0 = local only (conv +
  cond, the P0 behaviour), 1 = coarse radiative-emission grid, 2 = bounded gather over burning elements. PRD 8.2
  option 4 (visibility-aware) is deferred and documented in ADR-0006, not implemented. Because it lives in
  `sim.yaml`, it is captured by `app/replay.py::config_snapshot` and overridable by the test helpers that widen
  ranges. Task 1 ships with default 0 (no behaviour change); Task 2 picks the default by measurement.
- P2R2 **Radiation physics (PRD 8.2 formula, with two stated refinements in ADR-0006).** Emitters are burning
  elements b with temperature `T_b = max(T_s_b, T_rad_flame)` (the attached flame, artistic) and emissive power
  `w_b = eps_f * sigma * T_b^4 * A_b` [W]. Absorbed flux at receiver i:
  `q_rad_i = eps_i * sum_b w_b * c_i * c_b / (pi * max(r, rad_r_min)^2)` over emitters with `r <= rad_R`, where
  `c_b = |cos th_b|` (flames envelop the element) and `c_i = |cos th_i|` if the receiver's material is
  `consumed_when_burnt` (thin two-sided sheets: paper, cloth) else `max(cos th_i, 0)` (outward normal). `eps_i`
  (Kirchhoff absorptivity) is the refinement over the PRD formula. Self-term excluded (b != i). `q_rad` [W/m^2] is
  computed once per `SurfaceFireModel.step` (before the thermal sub-steps) and added to `q` in `thermal_kernel` for
  all elements (burning or not, flammable or not: non-flammables heat up but never ignite).
- P2R3 **Energy budget (extends ADR-0003).** Radiated power is not created: the total absorbed power
  `P_rx = sum_i A_i q_rad_i` is deducted from the emitters' gas share `heat_W_b` in proportion to `w_b`, clamped at
  `heat_W_b >= 0`; any clamped remainder is accumulated in a counter `rad_deficit_J` reported by `stats()`. Released
  power for the `flame` field (`released_power_W`) is unchanged.
- P2R4 **Emission grid resolution is independent of G.** Mode 1 bins emitter power on its own grid of cell size
  `rad_cell` [m] over the snapshot ROI (power-weighted centroid and mean normal per cell); receivers gather over the
  active cells as point emitters. It must not use `h_gas`, so mode 1 does not contaminate the resolution invariant.
- P2R5 **Determinism.** Emitter lists are compacted in element-index order with `wp.utils.array_scan` over the
  burning flags (no atomic-counter ordering); per-receiver sums run in list order. Mode 1 cell accumulation uses
  `wp.atomic_add` like the gas source scatter and relies on `wp.config.deterministic` (set in `sim/device.py`).
  CPU replay digests stay exact.
- P2R6 **Stage list unchanged.** Radiation runs inside the existing `surface` stage (`SurfaceFireModel.step`);
  `FireSimulation.stages()` keeps `["sample_gas", "surface", "gas"]` (asserted by
  `tests/test_coupled.py::test_step_equals_manual_stages`). Timing of radiation alone uses the public
  `SurfaceFireModel.compute_radiation()`.
- P2R7 **Workload version.** Changing the default `rad_mode` changes every coupled benchmark workload: Task 2 bumps
  `bench/workloads.yaml: workload_version` to 2 in the same commit and notes it in ADR-0006. The P0 baseline CSV stays
  as the version-1 record; the GPU baseline will be rerun by the user.
- P2R8 **Material status (PRD 8.1).** `configs/materials.yaml` keeps the PRD 8.5 per-material `status`. Task 2 adds
  an *additive* optional key `status_detail: {thermal: <s>, ignition: <s>, combustion: <s>, radiation: <s>}` per
  material (groups: thermal = rho, c, k, delta; ignition = T_ign, t_delay; combustion = m0_areal, mdot_ref, dH, chi,
  char_yield, y_soot, consumed_when_burnt; radiation = eps), validated by `MaterialTable` when present (values in
  `{physical_estimate, artistic}`), and records the per-group sources in `docs/NUMERICAL_METHODS.md`. No value is
  labelled `physical_estimate` without a source note ("order of magnitude from memory, verify" is an acceptable note;
  presenting it as measured is not).
- P2R9 **M9 monotonicity uses a parameter partial order, not a hard-coded material ranking.** Material A must ignite
  no later than B (same geometry, same exposure) when A's parameters are all no worse:
  `T_ign_A <= T_ign_B`, `C''_A <= C''_B`, `t_delay_A <= t_delay_B`, `(k*delta)_A <= (k*delta)_B`, `eps_A >= eps_B`,
  with at least one strict. With the current table this orders paper < cloth, paper < wood, paper < plastic,
  cloth < wood, cloth < plastic; **wood vs plastic is unordered** (wood has lower `T_ign` but higher `C''` and
  `t_delay`) and is reported, not asserted. The pairs are computed from `MaterialTable`, so retuning the table updates
  the assertions. Non-ignition within `t_max` counts as `+inf`; a violation is `t_A > t_B + dt_sim`.
- P2R10 **Resolution sensitivity (PRD 8.6).** Reference: the cloth patch of `table_v1` with the upper half of the
  paper lit (existing `cloth_response` helper in `tests/test_coupled.py`), metric = first cloth ignition time.
  A `slow` test gates G=32 vs G=64 at < 15 % (removing the P0 `xfail`); G=64 vs G=128 (the PRD pair) is measured once
  in this container by `eval/p2_resolution.py` run in the background (~6 s/step at G=128 on CPU) and committed as
  CSV; it is also in the user runbook. Any tuning loop runs at 32/64 only. If the gate cannot be met by tuning
  artistic parameters within their ranges, stop and report (do not widen the tolerance).
- P2R11 **Scenario snapshots without cameras.** Scripted and randomized scenarios are built from analytic primitives
  (`scene/synthetic.py` `Box`, `Rect`, `SyntheticScene.sample_surface`) binned with `scene/surfels.py::bin_surfels`
  and `build_neighbors`, with `cameras={}`, `view_depth={}`, `calib_version=0` (as the `grid_graph` helper in
  `tests/test_surface_invariants.py` does). Wind needs the gas solver, so scenarios run coupled. A small G over the
  3 m ROI is too coarse (G=32 -> h = 9.4 cm), so scenarios use their own ROI from `configs/scenarios.yaml`: a 0.96 m
  cube with G=32 -> h = 3 cm (cost of a G=32 step, ~1/8 of the G=64 CPU cost).
- P2R12 **Uniform-material ablation (PRD 11.2).** "Uniform" = every non-table object of the scene uses one flammable
  material whose parameters are the arithmetic mean of the four flammable classes (`consumed_when_burnt: false`),
  built in memory as a `MaterialTable` from edited entries (the 8-class order is kept; the uniform material replaces
  each object's class via `SurfaceFireModel.set_material_ids`). The table stays `other_nonflammable`.

---

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

## Task 3: P2-003 Scripted and randomized scenarios, M9 monotonicity eval, uniform-material ablation

**Files:** `src/livefire/sim/scenarios.py` (new), `src/livefire/scene/builder.py` (helper
`snapshot_from_primitives(...)`, no change to existing functions' behaviour), `configs/scenarios.yaml` (new),
`eval/p2_m9_spread.py` (new), `eval/p2_ablation_uniform.py` (new), `tests/test_scenarios.py` (new), Makefile
`eval-m9`, `eval-ablation-uniform` targets + README rows.

- `snapshot_from_primitives(objects: list[Box | Rect], grid_cfg: dict, seed: int, material_table=None) ->
  SceneSnapshot` (P2R11): samples each object with `SyntheticScene.sample_surface` (density so that each
  `surfel_spacing` cell gets >= 4 samples), bins with `bin_surfels` (`ViewPoints` with `footprint = area / n`,
  `has_normal` all True), neighbours with `build_neighbors` (same `neighbor_radius_factor`/`neighbor_k_max`), solids by
  the same thick-object rule as `snapshot_from_surfels` (reuse its code via a shared private function; do not
  duplicate), `cameras={}`, `calib_version=0`, an `ObjectRegistry` with `manual_material` per object.
- `configs/scenarios.yaml`: `grid: {G: 32, roi_min: [-0.48, -0.48, -0.05], roi_max: [0.48, 0.48, 0.91],
  surfel_spacing: 0.015}`, `t_max_s: 20.0`, `dt_check_steps: 6`, `random: {n_dev: 12, n_heldout: 12, seed_dev: 100,
  seed_heldout: 900, gap_m: [0.02, 0.12], angle_deg: [0, 60], size_m: [0.06, 0.16], wind_m_s: [0.0, 0.8],
  ignite_radius_m: 0.04}`, `materials_compared: [paper, cloth, wood, plastic]`, `scripted: [...]` (below).
- `sim/scenarios.py`: `Scenario` dataclass (`name, objects, ignite_point, ignite_radius, wind_speed, target_obj,
  seed`); `scripted_scenarios(cfg)` = (a) `demo_chain`: paper sheet, cloth 3 cm away, wood block 4 cm beyond, metal
  cup, glass and ceramic boxes nearby (PRD 13 demo order; a demo script, not a validation, PRD M9);
  (b) `two_sheets_parallel`, (c) `sheet_over_block`; `random_scenarios(cfg, split)` = seeded emitter paper sheet +
  target sheet with gap/angle/size/wind drawn from the ranges (dev and held-out seeds disjoint);
  `run_scenario(sc, target_material, params, app_cfg) -> dict` (coupled `FireSimulation`, wind via a sim.yaml
  override of `wind_speed`, checks the target every `dt_check_steps`, returns `t_ignite_s` or None, burnt fraction
  per object, max `T_g`, stats); `ordered_pairs(table) -> list[tuple[str, str]]` per P2R9.
- `eval/p2_m9_spread.py [--split dev|heldout|both] [--quick] [--grid G]`: for every scenario and every material in
  `materials_compared` set on the target only (same geometry, same emitter), run and record
  `eval/results/m9_spread_<date>.csv` (`scenario, split, seed, gap_m, angle_deg, wind_m_s, material, t_ignite_s,
  ignited, wall_s`) and `eval/results/m9_summary_<date>.csv` (per ordered pair: n scenarios, violations, pass;
  unordered pairs listed with `asserted=false`). **M9 passes** iff zero violations on the held-out split and at least
  one target ignites in >= 75 % of held-out scenarios (non-degenerate). Exit code 0 on pass, 1 on fail. Estimate the
  full run's wall time from `--quick` first; it must fit ~90 min CPU (reduce `t_max_s` or G only via the config, and
  record the values used in the CSV header row `notes`). Run it in the background and commit the CSVs.
- `eval/p2_ablation_uniform.py`: `table_v1` (G=32) and `demo_chain` with material-dependent vs uniform (P2R12)
  materials, same ignition: per object first-ignition time, burnt area at 10/30/60 s, total released energy, and
  "non-flammable object ignited" flag -> `eval/results/ablation_uniform_<date>.csv`. Answers the PRD 11.2 question
  "does material-dependent combustion change the spread": the report states the measured differences, no claims beyond
  the CSV.
- Tests (`tests/test_scenarios.py`, fast unless noted): snapshot builder produces valid snapshots (contract checks
  pass, symmetric neighbours, solids only for thick materials, element spacing within 20 % of `surfel_spacing`);
  random scenarios are reproducible from the seed and dev/held-out sets disjoint; `ordered_pairs` on the current
  table returns exactly the five pairs of P2R9 and excludes (wood, plastic); a synthetic table where all parameters of
  plastic are lowered below wood makes (plastic, wood) appear; the M9 checker flags a hand-made violation and treats
  None as +inf; `eval/p2_m9_spread.py --quick` (1 dev scenario, 2 materials, `t_max_s` 5, G 16) runs and writes a CSV
  (mark `slow` if > 30 s); uniform table builder keeps 8 classes in order and its flammable entry equals the mean.

**Acceptance:** fast tests green; full M9 CSVs committed with the pass/fail verdict; ablation CSV committed.

## Task 4: P2-004 Missing PRD 8.6 invariants and gas-solver quality metrics

**Files:** `tests/test_p2_invariants.py` (new), `src/livefire/sim/gas.py` (add `pressure_residual()` and
`boundary_penetration()` read-only diagnostics; no change to the step), `eval/p2_gas_quality.py` (new),
`docs/TEST_STRATEGY.md` (invariant table rows only), Makefile `eval-gas` + README row.

Already covered by P0 (do not duplicate; keep): fuel bounds (`test_surface_invariants.py::test_fuel_bounds_over_burn`,
also checks char/removed), non-flammables without radiation, no spontaneous fire (surface only), zero-wind gas symmetry
(`test_gas.py::test_zero_wind_symmetry`), conduction smoothing + energy (`test_conduction_energy_conservation`,
`test_stability_large_dt_substeps`), projection factor (`test_gas.py::test_projection_reduces_divergence_10x_default_iters`),
replay (`test_replay.py`, `test_coupled.py::test_replay_digest_equal_on_cpu`), long run (slow). Add only:

- With the Task 2 default radiation mode (read from `configs/sim.yaml`, and parametrised over modes 1 and 2):
  1. **No spontaneous fire, coupled:** `table_v1` G=32, no ignition, 120 steps with radiation on: nothing burning,
     `E == 0`, `T_s` within 1e-6 K of `T_amb`, `T_g` within 1e-4 K of ambient, `rad_emitters == 0`.
  2. **No spontaneous fire from sub-threshold heat:** surface model, a sheet exposed to gas at `T_ign - 20 K` for
     600 s never ignites.
  3. **Non-flammables under radiation:** surface model with `T_g` ambient: a metal, glass, ceramic and
     other_nonflammable receiver each 2 cm in front of a burning paper sheet for 60 s: heated (`T_s` rises > 1 K),
     never burning, `m_fuel`, `char`, `removed` stay exactly 0.
  4. **Char and removed in 0..1 with radiation spread** across `table_v1` for 60 s (surface only), and fuel bounds
     `m_res <= m_fuel <= m0` per element at 10 checkpoints.
  5. **Zero wind: no directional bias in spread or flame lean.** A square 0.3 m paper sheet (`snapshot_from_primitives`
     from Task 3 with the `configs/scenarios.yaml` grid, horizontal at z = 0.05 m, centred at x = y = 0 so the ROI is
     symmetric about it), coupled G=32, wind 0, ignited at the centre:
     after 2 s simulated the burnt-element centroid (area-weighted, `char > 0`) is within 0.5 `surfel_spacing` of the
     ignition point in x and y; the burnt extents in +x/-x and +y/-y differ by <= 1 spacing; the `flame` field
     centroid (x, y) is within 0.5 `h_gas` of the ignition point. Mark `slow` if > 30 s.
  6. **Replay with radiation:** two runs with the same seed, events and `rad_mode` give equal `FireStateHost.digest()`
     (CPU), and a one-element ignition-point change gives a different digest.
  7. **Projection with radiation-driven flow:** after 60 steps of a burning `table_v1` (G=32), the last step's
     `stats()["div_after"] < stats()["div_before"]`; print the factor. The PRD "stated factor" stays the P0 10x test on
     a random field; on smooth buoyant flow the factor is unmeasured, so this test asserts only `factor >= 0.8 *` the
     value you measure once while writing it (state the number in the test's docstring and the report; if it is below
     2, report it to the controller as a solver-quality finding rather than tuning the test).
- `GasSolver.pressure_residual() -> float`: `||div(u*) - (dt/h^2) lap(p)||_2 / ||div(u*)||_2` over fluid cells for the
  last projection (store what is needed during `_project`; syncs). `GasSolver.boundary_penetration() -> dict`:
  max |face velocity| over fixed faces excluding the -x inflow wall, and the max |net flux| into solid cells, after the
  last projection [m/s]. Both are diagnostics only; `step()` results must be bit-identical before/after this change
  (test: digest of a 10-step run equals the digest computed with the diagnostics never called).
- `eval/p2_gas_quality.py [--grids 32 64] [--iters 20 60 120]`: on `table_v1` with burning paper, per (G, n_iters):
  `||div||_2` before/after, ratio, relative residual, boundary penetration, velocity/temperature clamp counts, max
  CFL `|u| dt / h`, over 60 steps (median and max) -> `eval/results/gas_quality_<date>.csv`; also time the
  projection (CPU-labelled). Tests: the residual decreases monotonically with n_iters on a random field (20 < 60 < 120);
  penetration < 1e-5 m/s after projection with a solid box; the eval script `--grids 16 --iters 10` runs.
- Update the invariant table in `docs/TEST_STRATEGY.md` with the new tests (rows only).

**Acceptance:** fast tests green (slow ones run by node id once, results in the report); gas quality CSV committed.

## Task 5: P2-005 Resolution sensitivity and 10-minute long-run stability (radiation on)

**Files:** `tests/test_coupled.py` (replace the P0 `xfail` resolution test with the P2R10 gated test; the long-run
test body stays, it now runs with the radiation default), `tests/test_p2_longrun.py` (new: stress long run),
`eval/p2_resolution.py` (new), `configs/sim.yaml` (only if tuning is needed, within ranges), Makefile
`eval-resolution` + README row.

- Resolution (P2R10): `test_resolution_sensitivity_g32_vs_g64` (`slow`, no xfail): cloth first-ignition time with the
  paper lit at G=32 and G=64; assert both ignite within 20 s simulated and
  `|t32 - t64| / t64 < 0.15`. If it fails, find the root cause first (the `h_gas` sampling/injection offset of
  `SurfaceFireModel.sample_points`, gas heating near the sheet, radiation mode 1 binning) and tune only artistic
  parameters inside their ranges, documenting each change; a structural change (e.g. a fixed metric sampling offset
  instead of `h_gas`) needs an ADR note appended to ADR-0006 and controller approval before committing.
- `eval/p2_resolution.py [--grids 32 64 128] [--seconds 20]`: same reference scenario per G -> `eval/results/
  resolution_<date>.csv` (`G, h_gas_m, t_cloth_ignite_s, heat_10K_s, wall_s, steps`) plus the relative differences
  for (32, 64) and (64, 128), each with `pass = rel_diff < 0.15` (the PRD 8.6 verdict for the 64/128 pair, which the
  phase report uses). Write each G's row as soon as that G finishes. Run `--grids 64 128` in the background once
  (~5.9 s/step at G=128 on CPU: 20 s simulated = 1200 steps ~ 2 h; stop each run at cloth ignition, which shortens
  it) and commit the CSV; if G=128 cannot finish, commit what finished and list the 64/128 verdict as "pending user
  run" (the script prints the one-line command).
- Long run (PRD 8.6, 10 min simulated): the existing `test_long_run_10_min_bounded` (G=32, 36000 steps) runs unchanged
  with the radiation default; add to it the assertions `stats()["rad_overflow"] == 0` and `rad_deficit_J` finite.
  New `tests/test_p2_longrun.py::test_long_run_stress_wind_all_lit` (`slow`): G=32, `wind_speed=1.0`,
  `wind_noise=0.1`, every flammable object of `table_v1` ignited at t=0 and the paper re-ignition point hit again at
  t=300 s, 36000 steps: no NaN/Inf in any state array, `T_g <= T_g_max`, `|u|` within the clamp bound, fuel/char
  bounds, and "no unbounded growth": max `T_g`, max soot and total soot mass over the last 60 s are each <= their max
  over the first 300 s (fire has burnt out; growth would indicate instability). Run both long runs by node id in the
  background (~50 min each on CPU, measured in P0); record wall time and peaks in the report.

**Acceptance:** resolution test passes at 32/64 (or the task stops with a root-cause report); resolution CSV committed
(64/128 measured or explicitly pending); both long runs pass, results logged.

## Task 6: P2-006 Numerical-methods documentation, runbook rows and Phase 2 report (controller)

**Files:** `docs/NUMERICAL_METHODS.md`, `docs/reports/phase-2.md` (new), README "Windows (no make)" table check,
`docs/TEST_STRATEGY.md` (final pass).

- `docs/NUMERICAL_METHODS.md`: radiation model (P2R2-P2R5, ADR-0006), energy budget, stability statement for
  radiation, gas-solver quality (definitions of the metrics in Task 4 and where the CSV is), resolution behaviour
  (P2R10, measured numbers quoted from the CSV), scenario and M9 definitions (P2R9, P2R11), uniform ablation (P2R12),
  known limitations (no visibility/occlusion in radiation, constant `mdot_ref`, thin-sheet `|cos|` approximation,
  CPU-only evidence). Point to the PRD instead of restating it.
- `docs/reports/phase-2.md`: Gate 2 table with one row per criterion: every PRD 8.6 invariant (test node id, status,
  date run, CPU), with the resolution row showing both the gated 32/64 test and the PRD 64/128 verdict from
  `eval/results/resolution_<date>.csv` (or "pending user run"), M9 (summary CSV, pass/fail), documented numerical behaviour (doc links), plus the radiation
  comparison, gas quality, resolution and ablation results quoted from committed CSVs only. Recommended gate outcome
  (`PASSED`, `PASSED_WITH_APPROVED_DEVIATION`, `FAILED`) with reasons.
- **Pending user run** section (never fabricated): radiation cost on the RTX 4060 (`make bench-radiation`), GPU
  baseline with `workload_version: 2` (`make bench`), `make test-slow` on the GPU (both long runs; deterministic Warp
  atomics on CUDA for replay), resolution G=64 vs 128 if not finished here (`make eval-resolution`), M9 on the GPU at
  G=64 (optional confirmation). One command per item, Windows form included.
- Check that every P2 Makefile target has its README "Windows (no make)" row.

**Acceptance:** report numbers trace to committed CSVs or test node ids; pending items listed explicitly.
