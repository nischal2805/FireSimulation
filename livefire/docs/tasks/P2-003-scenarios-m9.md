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

