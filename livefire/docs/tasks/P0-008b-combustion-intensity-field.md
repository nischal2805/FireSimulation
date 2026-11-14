## Task 8b: P0-008b Combustion-intensity field for visible flames (ADR-0004)

**Why (controller ruling, measured):** with default parameters the coupled sim heats the gas near burning paper to
only 376 K at G=64 (390 K peak), far below the 700 K emission threshold: a thin flame averaged over a 4.7 cm cell
is warm air. PRD 7.4.3 says the flame appearance uses "local combustion intensity" separately from the solver.
P0 therefore adds a minimal combustion-intensity scalar to the gas state and lets the renderer use it.

**Files:** `src/livefire/sim/gas.py`, `src/livefire/sim/coupled.py`, `src/livefire/sim/state.py` (FireStateHost gains
`flame`), `src/livefire/common/interfaces.py` (FireStateView gains `flame`), `src/livefire/render/volume.py`,
`src/livefire/render/compositor.py`, `configs/sim.yaml`, `configs/render.yaml`, `tests/test_flame_field.py`,
`docs/decisions/ADR-0004-combustion-intensity-field.md`, `docs/PRD.md` (change log + 6.2 field list).

- New cell-centred field `flame` f32 (G,G,G), units W/m^3 (heat-release-rate density proxy). Per step, after the
  surface step: scatter each element's total released power `P_i = chi * dH * consumed * A / dt` (feedback + gas
  parts) trilinearly at the element sample points as `flame += P_i / h^3 * (1 - exp(-k_flame dt))` (same scatter
  helper and weights as the heat source). Then advect `flame` with the velocity (same semi-Lagrangian as T_g and soot)
  and decay `flame *= exp(-k_flame dt)`. Solid cells hold 0. With a steady source the cell value converges to the
  local heat-release-rate density, which is why the source carries the `(1 - exp(-k dt))` factor.
  `SurfaceFireModel` must expose the total released power per element (add a read-only accessor; do not change its
  existing return values). `sim.yaml`: `k_flame 6.0 1/s [0.5, 50] artistic` (flame lifetime ~0.17 s).
- Renderer: visual temperature `T_vis = max(T_g, T_amb + dT_flame * (1 - exp(-flame / flame_ref)))`, used for emission
  colour and intensity instead of T_g; absorption by soot unchanged; world-space only. `render.yaml`:
  `dT_flame 1100 K [0, 2500] artistic`, `flame_ref 1.0e6 W/m^3 [1e3, 1e9] artistic`. The renderer still never writes
  sim state. The active-box test (cells worth marching) must include `flame > 0.01 * flame_ref`.
- ADR-0004 (Appendix B format, accepted): context with the measured numbers above, the field definition, why it is a
  passive scalar and does not feed back into heat transfer (spread stays with radiation options in P2), consequences
  (resolution dependence of `flame` per cell; P4 appearance model refines it). PRD change-log entry v2.2 and a one-line
  addition to the 6.2 field list.
- Tests (`tests/test_flame_field.py`, G<=32, fast): (1) no fire -> flame identically 0 for 200 steps; (2) a lit paper patch
  -> flame > 0 above the patch within 0.5 s, flame >= 0 everywhere, solid cells 0; (3) steady-source check on a
  1-cell source with zero velocity: flame converges to `P / h^3` within 2 %; (4) after burnout (force m_fuel to m_res)
  flame decays below 1e-3 of its peak within 2 s; (5) rendering a coupled G=32 state 1 s after ignition gives visible
  emission (`L_vol` luminance sum > 0 within 20 px of the projected ignition point) in both views, and with
  `dT_flame = 0` the emission there is ~0 (proves the field drives visibility); (6) replay digest still equal with
  `flame` included in `FireStateHost.digest`.

**Acceptance:** tests pass; `make test` green; ADR-0004 and PRD change log committed.

