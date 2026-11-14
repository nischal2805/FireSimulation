## Task 1: P0-001 Repo skeleton, environment, config loader, timers

**Files:** `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `.gitignore`, `configs/{app,grid,camera,sim,render,materials}.yaml`,
`src/livefire/__init__.py`, `src/livefire/{common,capture,perception/depth,perception/segment,perception/material,fusion,scene,sim,render,app}/__init__.py`,
`src/livefire/common/{config.py,timers.py,log.py}`, `tests/test_config.py`, `tests/test_timers.py`, `bench/{results,plots,traces}/.gitkeep`, `eval/.gitkeep`.

- `pyproject.toml`: name `livefire`, src layout, `requires-python = "==3.11.*"`. Runtime deps: numpy, scipy, pyyaml,
  opencv-python-headless, warp-lang, matplotlib. Optional group `spike`: taichi. Dev group: pytest, ruff.
  pytest config registers marker `slow`. Ruff line length 110.
- `uv lock` and `uv sync --all-groups` must work; commit `uv.lock`.
- `.gitignore`: `data/`, `.venv/`, `__pycache__/`, `out/`, `*.mp4` under `out/`, `.superpowers/`, `bench/traces/*` (keep .gitkeep).
- Makefile targets: `setup` (`uv sync --all-groups`), `test`, `test-slow`, `lint` (`uv run ruff check src tests bench`),
  `run`, `bench`, `bench-spike`, `plots`, `calibrate`, `eval`. Targets for later tasks call fixed paths:
  `run` -> `uv run python -m livefire.app run --scene table_v1 --seconds 8 --ignite "-0.075,0.0,0.05@0.5" --out out/demo`,
  `bench` -> `uv run python bench/run_bench.py`, `bench-spike` -> `uv run python bench/spike/run_spike.py`,
  `plots` -> `uv run python bench/plots.py` (these files do not exist yet; that is expected); `calibrate` and `eval` print
  `"not implemented until P1"` and exit 2.
- `configs/app.yaml`: `seed: 1234`, `device: "cpu"`, `dt_sim: 0.0166666667` (1/60 s), `max_substeps_per_frame: 4`,
  `headless: true`, `log_level: INFO`.
- `configs/grid.yaml`: `G: 96`, `roi_min: [-1.5, -1.5, -0.1]`, `roi_max: [1.5, 1.5, 2.9]`, `surfel_spacing: 0.015`,
  `neighbor_radius_factor: 1.5`, `neighbor_k_max: 16`.
- `configs/camera.yaml`: synthetic rig: width 1280, height 720, fx=fy=900.0, cx=639.5, cy=359.5, dist zeros(5),
  `calib_version: 1`; cam0 position [1.0, -0.9, 0.7], cam1 position [-0.6, -1.1, 0.65], both target [0.0, 0.0, 0.08].
- `configs/materials.yaml`: PRD 8.5 table verbatim (with the units comment).
- `configs/sim.yaml` and `configs/render.yaml`: create with a header comment describing the
  `{value, unit, range, status}` form and one example-free empty mapping (`{}`); Tasks 7, 8, 9 add their parameters.
  The loader must accept an empty file/mapping.
- `common/config.py`: `load_yaml(name) -> dict` (from `configs/<name>.yaml`, root found relative to the package or via
  env `LIVEFIRE_CONFIG_DIR`); `Params` wrapper: `params = load_params("sim")`, `params["h_c"] -> float` returns `value`,
  validates every entry has `value, unit, range, status` with status in `{physical_estimate, artistic}` and
  `range[0] <= value <= range[1]`; raises `ConfigError` with the key name otherwise. `Params.meta(key)` returns the dict.
- `common/timers.py`: `StageTimer` using `time.perf_counter_ns`; `with timer.stage("name"):`; `timer.record(name, ms)`;
  `timer.summary() -> dict[name, {"n", "median_ms", "p95_ms", "p99_ms"}]`; `timer.reset()`. A `device_sync`
  callable can be passed to `stage(...)` to synchronise before reading the clock (GPU boundary rule PRD 5.5).
- `common/log.py`: `get_logger(name)` standard logging with level from app.yaml.
- Tests: config loads all six files; a bad status, missing unit, out-of-range value each raise `ConfigError`;
  timer percentiles are right on a known list (`record` 1..100 ms -> median 50.5, p95 ~95.05 using numpy percentile
  linear); `stage` measures a `time.sleep(0.01)` as >= 9 ms.

**Acceptance:** `make setup && make test && make lint` green.

