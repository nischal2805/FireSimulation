# ADR-0006: Radiative heat transfer between surface elements (`rad_mode`)

Date: 2026-11-15. Status: **accepted** for the numerics; GPU cost is a **pending user run** (measurement, not an
open decision).

## Context

HANDOVER risk 3: with local conduction and convection only (the P0 behaviour), fire does not spread between
objects (burning paper does not light the cloth next to it). PRD 8.2 asks to implement and compare, in this order:
(1) local only, (2) coarse radiative-emission grid with interpolation, (3) bounded gather over burning elements,
(4) optional visibility-aware radiation, and to pick by visual effect and measured GPU cost, the simplest
sufficient option winning. The visual target is a plausible fire, not a calibrated one (ADR-0003).

## Candidates (source, date, license, size)

All implemented in this repository (own code, Warp, same license as the project), P2-001, `src/livefire/sim/radiation.py`,
selected by `configs/sim.yaml: rad_mode` (P2R1):

0. `rad_mode 0`, PRD option 1: local only (conv + cond). Zero extra code path.
1. `rad_mode 1`, PRD option 2: emitters binned on a coarse grid (`rad_cell` = 0.05 m, independent of `G`, P2R4);
   receivers gather over the active cells as point emitters.
2. `rad_mode 2`, PRD option 3: each receiver gathers over the compacted list of burning elements within `rad_R`
   (cap `rad_max_emitters`).
3. PRD option 4, visibility-aware radiation: **deferred**. It needs per-pair occlusion (ray casts against the
   snapshot), which is a larger cost and implementation risk and is only worth it if the unoccluded result looks wrong
   in the user's scenes; nothing measured here says so. Not implemented.

Formula (P2R2, PRD 8.2 with two refinements): emitters are burning elements with `T_b = max(T_s_b, T_rad_flame)` and
`w_b = eps_f sigma T_b^4 A_b`; absorbed flux `q_i = eps_i sum_b w_b c_i c_b / (pi max(r, rad_r_min)^2)` for `r <= rad_R`.
Refinements: `eps_i` (Kirchhoff absorptivity) and `|cos|` for thin two-sided sheets (paper, cloth) and for the
emitters (flames envelop the element). Energy (P2R3): the absorbed power is deducted from the emitters' gas share
`heat_W`, clamped at 0, the clamped remainder being counted in `rad_deficit_J`.

## Method (data, metrics, hardware, commands)

`bench/p2_radiation_compare.py` (`make bench-radiation`), scene `table_v1`, default grid from `configs/grid.yaml`
(8292 surface elements), CPU only (Warp CPU device, 4 cores, Warp runs single threaded), 1/60 s steps (`dt_sim`).
Same paper ignition for every mode.

- Cost (CSV 1): median/p95 of `SurfaceFireModel.compute_radiation()` (stage `radiation`) and of one full surface
  step (stage `surface_step`) with T_g = T_amb, 5 repeats, 2 warm-up launches discarded, 10 (radiation) or 6
  (step) samples per repeat. Fire sizes: ignition radius 0.03, 0.08, 0.15 m and "all flammables" (10 m). The brief
  expected ~100 / ~1000 / ~3000 burning elements; `table_v1` has too few flammable elements near the ignition
  point (280 paper, 288 cloth, a few hundred wood), so the measured counts are 28 / 176 / 354 / 862 (in the variant
  name `mode<k>_burning<n>`).
- Effect (CSV 2): first ignition time and burnt area (elements that were ever burning) per object at 10 / 30 / 60 s,
  for two ignitions: `paper_all` (every paper element lit, the Task 1 spread scenario) and `local_r0.06` (the
  `make run` ignition: point (-0.075, 0, 0.1), radius 0.06 m), each on the surface model alone with the gas at
  ambient (`surface`, isolates radiation) and coupled at G=32 (`coupled_g32`, end to end).
- Run: `uv run python bench/p2_radiation_compare.py` in the background, **total wall time 32.9 minutes** (cost part
  0.1 min; the 6 coupled runs of 60 simulated seconds take about 5.3 min each) on this CPU container.
- Files: `bench/results/radiation_cpu_2026-11-15.csv` (Appendix C, `workload_version` 2, git sha of the run in the
  rows), `eval/results/radiation_effect_2026-11-15.csv`.

## Results (table, CSV links)

Cost, CPU (ms, median / p95), `bench/results/radiation_cpu_2026-11-15.csv`:

| burning elements | mode 0 radiation | mode 1 radiation | mode 2 radiation | mode 0 surface step | mode 1 surface step | mode 2 surface step |
| --- | --- | --- | --- | --- | --- | --- |
| 28 | 0.006 / 0.009 | 1.99 / 2.35 | 1.61 / 1.72 | 0.52 / 0.64 | 3.07 / 3.48 | 1.72 / 1.86 |
| 176 | 0.006 / 0.008 | 3.20 / 3.45 | 5.77 / 7.66 | 0.51 / 0.61 | 3.75 / 4.05 | 6.45 / 9.12 |
| 354 | 0.007 / 0.009 | 3.65 / 3.98 | 10.68 / 15.46 | 0.52 / 0.74 | 4.31 / 4.95 | 12.12 / 17.41 |
| 862 | 0.007 / 0.010 | 5.35 / 7.29 | 27.73 / 42.10 | 0.59 / 0.77 | 5.47 / 7.46 | 28.89 / 30.91 |

Mode 2 grows roughly linearly with the number of burning elements (N receivers x E emitters); mode 1 grows slowly
(active cells). For reference, the P0 CPU baseline coupled step is ~0.66 s at G=64 (`baseline_cpu_2026-11-14.csv`),
so the mode 2 surface step at 862 burning elements adds about 4 % to it on CPU. **GPU cost: pending user run**
(`make bench-radiation` on the RTX 4060 machine); nothing here is an estimate of it.

Effect, `eval/results/radiation_effect_2026-11-15.csv` (simulated seconds of first ignition; "-" = never within 60 s;
burnt area in m^2 is identical at 10/30/60 s for every row because all burning ends within 10 s):

| model, ignition | mode | cloth first ignition | cloth burnt area | wood | metal | max cloth T_s (K) | max paper T_s (K) | `rad_deficit_J` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surface, paper_all | 0 | - | 0 | never burns, T_s max 293.2 | never burns | 293.2 | 900.0 | 0 |
| surface, paper_all | 1 | 2.50 s | 0.0066 | never burns, 299.1 K | never burns, 294.0 K | 1154.5 | 1331.1 | 1.03 |
| surface, paper_all | 2 | 3.68 s | 0.0048 | never burns, 298.7 K | never burns, 294.0 K | 803.4 | 900.0 | 0 |
| coupled G=32, paper_all | 0 | - | 0 | never burns | never burns | 327.5 | 900.0 | 0 |
| coupled G=32, paper_all | 1 | 2.47 s | 0.0069 | never burns | never burns | 1157.8 | 1331.5 | 0.98 |
| coupled G=32, paper_all | 2 | 3.60 s | 0.0050 | never burns | never burns | 811.3 | 900.0 | 0 |
| surface, local_r0.06 | 0 / 1 / 2 | - / - / - | 0 | never burns | never burns | 293.2 / 429.5 / 430.4 | 900.0 | 0 |
| coupled G=32, local_r0.06 | 0 / 1 / 2 | - / - / - | 0 | never burns | never burns | 310.7 / 436.0 / 436.9 | 900.0 | 0 |

Paper burnt area: 0.0326 m^2 (paper_all, 280 elements) and 0.0124 m^2 (local, 101 elements) in every mode: the
paper itself does not spread by radiation (coplanar elements have cos = 0) and conduction plus gas heating do not
spread it either within the scenario. Metal (a non-flammable) never burns in any run and the cloth/wood/metal
are heated by at most a few K when they are out of reach. Glass is not in `table_v1`; non-flammable behaviour
is covered by the unit tests (`tests/test_radiation.py`).

Observations that count against mode 1: (i) first ignition 1.2 s earlier than mode 2 (2.50 vs 3.68 s); (ii) burning paper reaches 1331 K in mode 1
versus 900 K (T_pilot, never exceeded) in modes 0 and 2, i.e. burning elements receive spurious flux from their own
neighbourhood, a consequence of the cell-centroid approximation within about one `rad_cell` of an emitter (Task 1
tests: L-corner near field); (iii) `rad_deficit_J` is non-zero (about 1 J) in mode 1: the emission demanded
more than the gas share could give; mode 2 stays at 0. The cause of (ii) and (iii) was attributed to the near-field
inaccuracy, not isolated further.

## Scores (11.6, adapted to a technique choice)

0-5 per criterion; weights from PRD 11.6 where they apply (accuracy 35 as "visual effect", latency/VRAM 30 as cost,
robustness 20, implementation risk 10; the licence criterion (5) is the same for all candidates and omitted).

| criterion (weight) | mode 0 (local) | mode 1 (grid) | mode 2 (gather) |
| --- | --- | --- | --- |
| visual effect (35): cloth ignites from burning paper | 0 (never, CSV 2) | 3 (ignites, 2.5 s, too early, paper over-heats) | 4 (ignites, 3.7 s, paper bounded) |
| cost now (30): CPU table, GPU "pending user run" | 5 (0.007 ms) | 4 (5.3 ms at 862) | 3 (27.7 ms at 862, linear in emitters) |
| robustness (20) | 5 | 2 (near-field error, deficit, 2e6-cell cap) | 4 (exact within `rad_R`, bounded by `rad_max_emitters`, overflow counted) |
| implementation risk (10) | 5 | 2 (atomic float sums, binning) | 4 |
| weighted total (max 475) | 300 | 285 | 350 |

Scores are the author's judgement from the tables above; the CPU cost is not the GPU cost. Mode 0 is eliminated
by the hard requirement (cloth must ignite from burning paper) regardless of its total.

## Decision and fallback

**Default `rad_mode = 2`** (bounded gather). Mode 0 is insufficient: the cloth never ignites (CSV 2). Both
modes 1 and 2 are sufficient in the narrow sense of lighting the cloth, so the tie-break "simplest sufficient wins"
(PRD 8.2) was read as simplest by implementation risk and correctness rather than by the order of the list: mode 2
is a direct sum with no grid, while mode 1 adds a binning approximation and shows the artefacts (i)-(iii)
above. If the reviewer reads "simplest" as list order, the default is `rad_mode 1` by changing one number; this is
the stated fallback.

**Fallback = mode 1** (`rad_mode: 1` in `configs/sim.yaml`), to be used if the GPU run shows that mode 2 is too
slow at the largest fires (the cap `rad_max_emitters` = 8192 bounds it but N x E can be large), and to be
re-examined for its near-field error then. No change to the tests is needed to switch (they set the mode explicitly).

Tuned artistic values: **none**. `eps_f` 0.7, `T_rad_flame` 1100 K, `rad_R` 0.30 m, `rad_r_min` 0.01 m,
`rad_cell` 0.05 m, `rad_max_emitters` 8192 keep the Task 1 values (declared ranges unchanged).
Known visual consequence: the demo ignition (`local_r0.06`) does **not** spread to the cloth within 60 s in any mode (cloth heats
to about 430 K, `T_ign` 520 K), whereas lighting the whole paper sheet does. Raising `eps_f` or `T_rad_flame`
within their ranges would make the local ignition spread; this is a look decision left to the controller/user.

## Consequences

- P2R3: radiated energy is taken from the emitters' gas share, so the coupled gas sees slightly less heat while
  an object is irradiated; counted in `rad_deficit_J` when the gas share is exhausted (zero in mode 2 here).
- `bench/workloads.yaml: workload_version` is 2 (P2R7): the default mode changes every coupled benchmark
  workload; `baseline_cpu_2026-11-14.csv` stays as the version 1 (no radiation) record and the GPU baseline is to be
  rerun by the user.
- The resolution invariant (PRD 8.6) uses the default mode 2. Mode 2 does not use `h_gas`; mode 1 is also
  independent of `G` by construction (P2R4).
- `tests/test_thermal_audit.py` checks the lumped thermal model against closed forms (radiative heating of a
  receiver and convective relaxation) and the unit/status audit of `configs/sim.yaml`; materials carry
  `status_detail` (P2R8).
- Limitations carried from P2-001: no occlusion (a burning object heats anything within `rad_R` regardless of
  what is between); mode 1 is inaccurate within about one `rad_cell` of an emitter; the emitter set is the
  `burning` flags at the start of the step; emission is bounded only through `rad_deficit_J`, not by the heat
  release; point pairs closer than 1e-9 m (`r^2 <= 1e-18`) are skipped; on a GPU the float atomic sums of mode 1
  (cell binning, `p_rx`, `sum_w`, deficit) rely on Warp's run-to-run deterministic mode and are **unverified**
  (CPU results are exact).
- Pending user run (GPU): `make bench-radiation` on the RTX 4060 machine for the GPU cost of modes 1 and 2 and a
  check of GPU determinism; this ADR is revisited only if mode 2 does not fit the budget.
