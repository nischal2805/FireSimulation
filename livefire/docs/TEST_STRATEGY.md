# Test strategy

Spec: `docs/PRD.md` section 10 (test categories), section 8.6 (numerical invariants), section 11.4 (benchmark rules).
This file maps the existing test files to those items. It does not restate them.

## Commands

- `make test`: `uv run pytest -m "not slow" -q` (fast tests).
- `make test-slow`: `uv run pytest -q` (all tests, including the `slow` marker).
- `make lint`: `ruff check src tests bench`.

The `slow` marker is registered in `pyproject.toml`. It is used only in `tests/test_coupled.py`: the 10-minute
simulated long run (PRD 8.6) and the R9 resolution-sensitivity test, which is marked `xfail(strict=False)` because
it reports rather than gates in P0. The long run uses G=32 in this container; G=64 runs in `make test-slow` on the
GPU machine.

Tests are CPU-only. Warp runs on the `cpu` device. Taichi tests in `tests/test_spike_kernels.py` are skipped when
Taichi is not importable.

## Mapping to PRD 10

| PRD 10 item | Test files | Covers |
| --- | --- | --- |
| 1. Unit tests | `test_geometry.py`, `test_config.py`, `test_errors_config.py`, `test_timers.py`, `test_contracts.py`, `test_materials.py`, `test_lifecycle.py` | Geometry contract, config loading and validation, timers, interfaces, material table, lifecycle table |
| 1. Unit tests (kernels on tiny inputs) | `test_spike_kernels.py` | Warp (and Taichi, if present) advection kernels against a NumPy reference |
| 2. Deterministic synthetic fixture | `test_synthetic.py`, `test_builder.py` | Synthetic `table_v1` scene and camera source; snapshot build from the fixture |
| 3. Numerical tests (PRD 8.6) | `test_surface_invariants.py`, `test_gas.py`, `test_flame_field.py`, `test_coupled.py` | Fuel bounds, non-flammables, zero-wind symmetry, conduction, projection divergence, sources, flame field, coupled step |
| 4. Integration | `test_builder.py`, `test_render.py`, `test_coupled.py` | Snapshot matches fixture geometry; both views render the same fire state at its 3D position |
| 5. Replay | `test_replay.py`, `test_coupled.py` | Same seed and inputs give the same `FireStateHost.digest()`; override behaviour |
| 6. Benchmark | `test_bench_csv.py` | Appendix C CSV schema, row round-trip, required fields, CPU labelling |
| 7. Eval scripts | none in P0 | `make eval` is a P1 target |

Also in the suite and outside PRD 10's numbered list: `test_degradation.py` (degradation ladder, PRD 5.6) and
`test_render.py` for the volume renderer and compositor.

## Invariant coverage (PRD 8.6)

| Invariant | Where it is checked |
| --- | --- |
| Fuel bounds over burn | `test_surface_invariants.py::test_fuel_bounds_over_burn` |
| Non-flammables never ignite or consume fuel | `test_surface_invariants.py` (parametrised over `metal`, `glass`, `ceramic`, `other_nonflammable`, plus the real-scene case) |
| No spontaneous fire | `test_surface_invariants.py::test_no_spontaneous_fire` |
| Zero wind, no directional bias | `test_gas.py::test_zero_wind_symmetry` |
| Heat conduction stable and energy-conserving | `test_surface_invariants.py` (`test_conduction_energy_conservation`, `test_stability_large_dt_substeps`) |
| Projection reduces divergence | `test_gas.py` |
| Replay determinism | `test_replay.py`, `test_coupled.py` |
| Long run, no NaN or unbounded growth | `test_coupled.py` (`slow`) |
| Resolution sensitivity | `test_coupled.py` (`slow`, `xfail`, R9: reported, not gating in P0) |

## Benchmarks

Benchmark numbers are not test assertions. The harness (`bench/run_bench.py`, `bench/spike/run_spike.py`) writes CSVs
under `bench/results/`. See `docs/PERFORMANCE_BUDGET.md` for the paths and rules.
