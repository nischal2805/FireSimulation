# ADR-0004: Combustion-intensity field for visible flames

Date: 2026-11-14. Status: **accepted** (controller ruling, P0-008b).

## Context

With the default parameters the coupled sim heats the gas near burning paper to only 376 K at G=64 (390 K peak),
far below the 700 K emission threshold of the renderer: a thin flame averaged over a 4.7 cm cell is warm air, so
the renderer (which derives emission from `T_g`) shows no flame. PRD 7.4.3 says the flame appearance uses "local
combustion intensity" separately from the solver. P0 therefore adds a minimal combustion-intensity scalar to the
gas state and lets the renderer use it.

## Candidates (source, date, license, size)

1. Raise `T_g` (more heat per cell, lower `k_cool`): the gas then buoyantly accelerates and the thermal state is
   wrong for the sake of appearance; rejected.
2. Passive combustion-intensity scalar `flame` (chosen): small (one `G^3` float32 field), no coupling back.
3. Fuel/mixture-fraction or flamelet model: P4 appearance research; out of scope.

## Method (data, metrics, hardware, commands)

`tests/test_flame_field.py` (CPU Warp, G=32): no fire gives `flame == 0` for 200 steps; a lit patch gives
`flame > 0` above it within 0.5 s, `flame >= 0`, solid cells 0; a steady 1-cell source converges to `P / h^3` (error
6e-6 after 120 steps); after forced burnout `flame` falls below 1e-3 of its peak within 2 s; the renderer shows
emission driven by `flame` (and none with `dT_flame = 0`); the replay digest includes `flame`.
Command: `uv run pytest tests/test_flame_field.py`.

## Results (table, CSV links)

Measured 1 s after ignition: G=32, r = 0.03 m patch (13 elements): `flame` peak 3.7e4 W/m^3 (`T_g` peak 304 K);
G=32 whole paper sheet (r = 0.2 m): 4.9e5 W/m^3; G=64, r = 0.03 m: 2.1e5 W/m^3 (`T_g` peak 329 K). The first value
chosen for `flame_ref` (1e6 W/m^3, an estimate) was about 30x too high: the emission threshold of 700 K needs
`flame > 4.6e5` there, so no flame was visible at G=32 even for the whole sheet. Ruling: `flame_ref = 3e4 W/m^3`
(artistic), which gives `T_vis` of about 1070 K for the G=32 small patch and saturated flames at G=64.
Rendered with the shipped defaults (G=32, r = 0.03 m), luminance within 20 px of the projected ignition point
(cam0 / cam1): 8.7 / 79.9 with `dT_flame = 1100`, 0.12 / 0.22 with `dT_flame = 0` (the remainder is soot in-scatter,
not emission). `flame_ref` is resolution dependent and needs re-tuning when `G` changes.

## Scores (11.6)

Not scored (artistic appearance parameter).

## Decision and fallback

- New cell-centred field `flame` float32 `(G,G,G)`, W/m^3 (heat-release-rate density proxy). After the surface
  step, each element's total released power `P_i = chi * dH * consumed * A / dt` (feedback + gas parts, exposed by
  `SurfaceFireModel.released_power_W`) is scattered with the heat-source weights at the sample points as
  `flame += P_i / h^3 * (1 - exp(-k_flame dt))`; then `flame` is advected semi-Lagrangian with the gas velocity
  (as `T_g` and soot) and decays by `exp(-k_flame dt)` (applied before the step's source so that a steady source
  converges to `P / h^3`). Solid cells hold 0. `k_flame = 6 1/s` (artistic, lifetime about 0.17 s).
- It is a passive scalar for the renderer and does not feed back into heat transfer; flame spread stays with the
  radiation options of P2.
- Renderer: `T_vis = max(T_g, T_amb + dT_flame (1 - exp(-flame / flame_ref)))` (`dT_flame = 1100 K`, `flame_ref =
  3e4 W/m^3`, artistic) drives emission colour and intensity; soot absorption is unchanged; world space only; the
  renderer never writes sim state. The active box includes cells with `flame > 0.01 flame_ref`.
- Fallback: tune `flame_ref`/`dT_flame`; P4 replaces this with an appearance model.

## Consequences

- `flame` per cell depends on the grid resolution (the same power is spread over `h^3`), so `flame_ref` must be
  re-tuned when `G` changes.
- `FireStateView` and `FireStateHost` (and its digest) gain `flame`.
