# Architecture

Spec: `docs/PRD.md` sections 5 (system architecture), 6 (scene representation, dynamic state) and 9 (interfaces).
This file records only what the code adds. Decisions: `docs/decisions/` (ADR-0001 GPU framework, proposed;
ADR-0002 MAC grid; ADR-0003 energy split; ADR-0004 combustion-intensity field).

## Module map (`src/livefire/`)

| Package | Modules | Role |
| --- | --- | --- |
| `common/` | `config`, `timers`, `log`, `geometry`, `interfaces`, `errors`, `bench_csv` | YAML config loading, timers, geometry contract helpers (`CameraCalibration`, `AABB`, `VoxelGrid`), protocols, errors, benchmark CSV writer |
| `capture/` | `types`, `synthetic_source` | `Frame` type and the synthetic two-camera source (P0 has no real cameras) |
| `perception/` | `types`, `depth/`, `segment/`, `fusion/`, `material/types` | Perception types. `material/types.py` fixes the 8-class order (`MATERIAL_CLASSES`, index = `material_id`). `depth/`, `segment/`, `fusion/` are empty placeholders until P1 |
| `scene/` | `synthetic`, `snapshot`, `builder`, `surfels`, `splat` | Synthetic scene `table_v1`, immutable `SceneSnapshot`, snapshot builder, surface-element sampling, depth splatting for occlusion |
| `sim/` | `materials`, `state`, `device`, `surface`, `gas`, `coupled` | Material table, host state copy, Warp device helpers, surface-element model, MAC gas solver, coupled `FireSimulation` |
| `render/` | `blackbody`, `volume`, `overlay`, `compositor` | Blackbody colour map, volume ray-march, overlay, per-camera compositor |
| `app/` | `lifecycle`, `runner`, `hud`, `replay`, `degradation`, `build`, `__main__` | Lifecycle state machine, frame runner, HUD, record/replay, degradation ladder, wiring (`build.py`), CLI entry point |

## Lifecycle

`app/lifecycle.py` defines the forward path `UNINITIALIZED -> CALIBRATING -> ACQUIRING -> READY -> SIMULATING`,
with `SIMULATING <-> PAUSED`. Any working state can go to `ERROR`, and `ERROR` returns to `UNINITIALIZED`.
Transitions outside the explicit `ALLOWED` table raise
`IllegalTransition`. `DEGRADED` is a flag shown alongside `SIMULATING`, not a state. The table is tested in
`tests/test_lifecycle.py`.

`build.py` runs acquisition once: it builds the `SceneSnapshot` (`scene/builder.py`), then constructs
`FireSimulation` (`sim/coupled.py`) from it. Surface elements are re-sampled only at READY (PRD 6.1).

## Ownership

- **`SceneSnapshot`** is host NumPy data. Its arrays are set to `writeable = False` at construction
  (`scene/snapshot.py`). Nothing mutates it after READY.
- **Device mirrors** are created once from the snapshot by each backend that needs them. `FireSimulation.reset()`
  (`sim/coupled.py`) uploads them, and the sim and render backends own their mirrors. The PRD 6.1 comments say
  "on GPU"; this is the ownership statement that applies.
- **Dynamic state** lives on the device inside `SurfaceFireModel` (`sim/surface.py`) and `GasSolver`
  (`sim/gas.py`). `FireSimulation.state_host()` returns a `FireStateHost` (`sim/state.py`) that the caller owns.
  It is a copy, not an alias of device memory.
- **Calibration** is checked, not trusted: every camera's `calib_version` must match the snapshot's, and a
  mismatch raises (`assert_calib_version`).

## Field table (PRD 6.2)

Surface arrays are per element, shape `(N_e,)`, where `N_e` is the number of surface elements. Gas arrays use the
grid `G^3` from `configs/grid.yaml` (default `G = 96`). Internally, surface state is float64 and the gas solver is
float32. `FireStateHost` exports float32.

| Field | Shape | Units | Memory layout | Owner | Initialisation |
| --- | --- | --- | --- | --- | --- |
| `T_s` | `(N_e,)` | K | SoA, float64 on device, f32 on host | `SurfaceFireModel` | `T_amb` (`configs/sim.yaml`) |
| `ignition_exposure` | `(N_e,)` | s | `E` in device code | `SurfaceFireModel` | 0 |
| `m_fuel` | `(N_e,)` | kg/m^2 | areal, from material `m0_areal` | `SurfaceFireModel` | `m0` of the element's material (0 for non-flammable) |
| `char` | `(N_e,)` | 0..1 | | `SurfaceFireModel` | 0 |
| `removed` | `(N_e,)` | 0..1 | visual burn-away | `SurfaceFireModel` | 0 |
| `burning` | `(N_e,)` | u8 (0/1) | | `SurfaceFireModel` | 0 |
| `T_g` | `(G,G,G)` | K | cell-centred C-order, f32 | `GasSolver` | `T_amb` |
| `vel` | `(G,G,G,3)` | m/s, world axes | derived, cell-centred (accessor) | `GasSolver` | 0, plus wind inflow on -X face |
| `soot` | `(G,G,G)` | kg/m^3 | cell-centred | `GasSolver` | 0 |
| `p` | `(G,G,G)` | Pa on export | stored kinematically as `p / rho_g` (m^2/s^2) internally | `GasSolver` | 0 |
| `flame` | `(G,G,G)` | W/m^3 | cell-centred; combustion intensity (ADR-0004) | `GasSolver` | 0 |

Velocity is staggered internally (ADR-0002). The MAC face arrays are `u (G+1,G,G)` on `(i, j+.5, k+.5)`,
`v (G,G+1,G)` on `(i+.5, j, k+.5)` and `w (G,G,G+1)` on `(i+.5, j+.5, k)`, in cell units from `roi_min`.
`GasSolver.velocity_cell_centered()` averages them to the `(G,G,G,3)` cell-centred `vel` exposed in `FireStateHost`.

Solid and gas temperatures are separate fields (PRD 6.2). Solid cells are marked by a `solid` mask. Faces touching a
solid cell are no-through (fixed velocity 0), so solid cells carry no flow.

## Pointers

- Numerical methods and controller rulings: `docs/NUMERICAL_METHODS.md`
- Frames and conventions: `docs/COORDINATE_CONVENTIONS.md`
- Tests: `docs/TEST_STRATEGY.md`
- Performance: `docs/PERFORMANCE_BUDGET.md`
