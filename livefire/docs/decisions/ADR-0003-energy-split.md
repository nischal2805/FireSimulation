# ADR-0003: Combustion energy split, surface re-radiation and temperature clamps

Date: 2026-11-14. Status: **accepted** (plan ruling R5; implemented in P0-007 and P0-008).

## Context

PRD 8.2/8.3 define the heat-release rate per element but not where the energy goes. Without a rule the surface
model and the gas solver would double count it, and with no radiative loss and no upper bound the temperatures
run away (positive feedback surface <-> gas). The visual target is a plausible fire, not a calibrated one.

## Candidates (source, date, license, size)

1. All heat to the gas; surface heated only by convection. Simple, but no flame-to-fuel feedback, so spread
   depends only on the (weak, coarse) gas temperature.
2. Fixed split with surface re-radiation and clamps (chosen). The fraction returned to the fuel is an artistic
   parameter; no literature value is claimed or cited (PRD 0.2: no unverified citations).
3. Coupled radiation transport (P1+ research): out of scope for P0.

## Method (data, metrics, hardware, commands)

Implemented in `src/livefire/sim/surface.py` (split, re-radiation, `T_s` clamp) and `src/livefire/sim/gas.py`
(injection, cooling, `T_g` clamp). Verified by `tests/test_surface_invariants.py`, `tests/test_gas.py`
(source energy conservation to 1e-4 relative) and `tests/test_coupled.py::test_long_run_10_min_bounded` (10
simulated minutes, clamp counters reported).

## Results (table, CSV links)

Source injection test: `sum(dT_g) rho_g c_p h^3 == heat_W dt` to 1e-4 relative, with and without a solid next to the
element (weights renormalised over fluid cells). Long-run results are in `task-8-report.md`.

## Scores (11.6)

Not scored (artistic parameter; no competing measured candidate).

## Decision and fallback

- Release: `q_release = chi * dH * mdot''` per burning element (W/m^2 of surface, material table).
- Split: fraction `f_feedback` (`configs/sim.yaml`, artistic, default 0.3) heats the surface as a flux
  `f_feedback * chi * dH * mdot_ref`; the remaining `1 - f_feedback` is returned by `SurfaceFireModel.step` as
  `heat_W` and injected into the gas at the sample point `x_i + n_i h_gas` as `T_g += heat_W dt / (rho_g c_p h^3)`
  (trilinear, fluid cells only). Soot: `soot_kg = y_soot * burnt mass`, added as `soot_kg / h^3`.
- Surface re-radiation loss `eps * sigma * (T_s^4 - T_amb^4)` in the surface thermal update.
- Gas losses (artistic surrogate for radiation/entrainment not resolved by the grid): `T_g = T_amb + (T_g - T_amb)
  exp(-k_cool dt)`, `soot *= exp(-k_soot dt)`.
- Clamps, each counted: `T_s` in `[T_s_min, T_s_max]` (surface), `T_g <= T_g_max` (gas); velocity `|u| <= cfl_max h / dt`
  (before projection). Counters are exposed by `FireSimulation.stats()`.
- Fallback: if flame-to-fuel feedback is too weak/strong visually, tune `f_feedback`; a flame/radiation sub-model
  is a P1+ decision.

## Consequences

- Total released energy is conserved between surface and gas (split is exact; the energy removed by `k_cool`,
  re-radiation and clamps is an explicit, counted loss).
- Observed in the baseline: with the default parameters the gas heated by a small burning patch stays near
  300-360 K and does not ignite neighbouring materials (flame spread beyond the lit region is a tuning topic).
- Parameters are artistic; calibrating them to measured fire is not claimed.
