# LiveFire Previs

Live two-camera virtual-fire previs. All fire is virtual. See `docs/PRD.md` for the spec and `docs/STATUS.md` for progress.

P0 is a synthetic vertical slice (no AI, no cameras).

Repository: https://github.com/nischal2805/FireSimulation (this project lives in the `livefire/` subfolder of that repo).

## Setup

Requires Python 3.11 exactly and [uv](https://docs.astral.sh/uv/). The environment is locked by `uv.lock`.

```
make setup   # uv sync --all-groups (creates .venv, Python 3.11)
```

Run tools through `uv run` (the make targets do this for you).

## Commands

| Target | What it does |
| --- | --- |
| `make test` | fast tests (`pytest -m "not slow"`) |
| `make test-slow` | all tests, including the long-run and resolution tests |
| `make lint` | `ruff check src tests bench` |
| `make run` | offline demo on the synthetic `table_v1` scene (writes to `out/demo`) |
| `make bench` | benchmark harness, writes CSVs to `bench/results/` |
| `make bench-spike` | framework spike kernels benchmark |
| `make plots` | plots generated from the CSVs in `bench/results/` |
| `make calibrate`, `make eval` | placeholders, not implemented until P1 |

## CPU-only note

This P0 work was developed and tested on a machine with no GPU. Warp runs on its CPU device, selected by
`device: "cpu"` in `configs/app.yaml`. The GPU target is `device: "cuda:0"`, which is set by hand on the GPU machine.
Never hard-code `cuda` in code.

All benchmark numbers committed so far are CPU numbers and are labelled `gpu=none`. ADR-0001 (GPU framework) stays
*proposed* until the RTX 4060 run. See `docs/PERFORMANCE_BUDGET.md`.

## Documentation

- `docs/ARCHITECTURE.md`: module map, lifecycle, ownership, field table
- `docs/NUMERICAL_METHODS.md`: solver methods, ADRs, controller rulings, parameter files
- `docs/COORDINATE_CONVENTIONS.md`: geometry contract as implemented
- `docs/TEST_STRATEGY.md`: test files mapped to PRD section 10
- `docs/PERFORMANCE_BUDGET.md`: benchmark CSVs and VRAM status
- `docs/decisions/`: ADRs
