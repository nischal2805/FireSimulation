# LiveFire Previs

Virtual fire for film previsualization. Two cameras look at a real table scene; the system reconstructs the scene's
3D geometry and materials from the live frames, simulates fire spreading through the objects on the GPU, and
composites the same 3D fire into both camera views in real time. **All fire is virtual**: nothing here controls real
heat or flame.

- Spec: [`docs/PRD.md`](docs/PRD.md) · Rules for agents: [`CLAUDE.md`](CLAUDE.md) · Progress: [`docs/STATUS.md`](docs/STATUS.md)
- Handover notes (where the project is and where to go next): [`docs/HANDOVER.md`](docs/HANDOVER.md)

## Status

| Phase | What | State |
| --- | --- | --- |
| P0 | Contracts and a synthetic two-camera vertical slice (sim, gas solver, renderer, replay, benchmarks) | Done, Gate 0 approved |
| P1 | Real cameras: live capture, click-to-ignite, ChArUco calibration, depth, fusion, live mode | In progress: capture, click-to-ignite (synthetic scene), calibration, depth and acquisition are built; **live mode on real cameras (P1-006) is not built yet** |
| P2-P5 | Simulation quality, scene understanding, visual fidelity and performance, final evaluation | Not started |

Everything so far was developed and tested on a CPU-only Linux machine. Nothing has run on a GPU or on Windows yet;
treat the GPU and Windows paths as untested.

## What you can run today

| Feature | Command |
| --- | --- |
| **Burn a real scene (live cameras)** | `uv run --extra depth python -m livefire.app live` (needs calibration, see below) |
| Click on the synthetic scene to start fires | `uv run python -m livefire.app interactive --scene table_v1` |
| Offline two-view fire demo video | `uv run python -m livefire.app run --scene table_v1 --out out/demo --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.06"` |
| See your cameras (fps, frame age, skew, status) and record clips | `uv run python -m livefire.app capture` (add `--record data/clips/run1 --seconds 10`) |
| Print and use the calibration board | `uv run python -m livefire.app calib board --out board.png --dpi 300` |

## Setup

Requirements: Python 3.11 (exactly) and [uv](https://docs.astral.sh/uv/). Dependencies are locked in `uv.lock`.

```bash
git clone -b agent/inspiring-maxwell-5od2va https://github.com/nischal2805/FireSimulation.git
```

```bash
cd FireSimulation/livefire
```

```bash
uv sync --all-groups --all-extras
```

`--all-extras` installs the optional `spike` (Taichi) and `depth` (PyTorch + Transformers) extras. PyTorch comes from
the CUDA 12.8 wheel index on Windows and the CPU index on Linux (see `pyproject.toml`).

**GPU:** set `device: "cuda:0"` in `configs/app.yaml` (default `cpu`). Never hard-code `cuda` in code.

## Commands

Run from the `livefire/` folder. `make` targets exist for Linux/macOS; on Windows (no `make`) use the right-hand
column. No code path shells out to bash.

| Make target | Command (works everywhere) | What it does |
| --- | --- | --- |
| `make setup` | `uv sync --all-groups --all-extras` | create `.venv` and install everything |
| `make test` | `uv run pytest -m "not slow" -q` | fast tests |
| `make test-slow` | `uv run pytest -q` | all tests incl. the 10-minute-simulated stability run |
| `make lint` | `uv run ruff check src tests bench` | lint |
| `make run` | `uv run python -m livefire.app run --scene table_v1 --out out/demo --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.06"` | offline demo video `out/demo/composite.mp4` |
| `make run-rt` | `uv run python -m livefire.app run --scene table_v1 --out out/demo_rt --seconds 20 --ignite "-0.075,0.0,0.1@0.3:0.06"` | real-time run (needs a GPU to be real time) |
| `make live` | `uv run --extra depth python -m livefire.app live` | live cameras: acquire, then click to ignite |
| `make interactive` | `uv run python -m livefire.app interactive --scene table_v1` | click-to-ignite viewer |
| `make capture` | `uv run python -m livefire.app capture` | live camera check |
| `make capture-headless` | `uv run python -m livefire.app capture --headless --seconds 5` | camera check without a window |
| `make latency-timer` | `uv run python -m livefire.app latency-timer` | on-screen ms counter for glass-to-glass latency |
| `make calibrate` | `uv run python -m livefire.app calib board --out board.png --dpi 300` | print-ready board (other `calib` subcommands below) |
| `make bench` | `uv run python bench/run_bench.py` | benchmarks to `bench/results/*.csv` |
| `make bench-spike` | `uv run python bench/spike/run_spike.py` | Warp vs Taichi kernel spike (decides ADR-0001) |
| `make plots` | `uv run python bench/plots.py` | plots from the CSVs |
| - | `uv run python bench/depth_spike.py` | depth model spike (ADR-0005) |
| - | `uv run python -m livefire.app replay out/demo` | replay a recorded run and check its state digest |

## Cameras

Cameras are listed in [`configs/cameras.yaml`](configs/cameras.yaml) (one or two entries).

- **USB webcam:** `type: usb`, `source: 0` (device index). On Windows DirectShow is used; `fourcc: MJPG` is usually
  needed for 1280x720 at 30 fps.
- **Phone as IP camera:** install an IP camera app (Android "IP Webcam": `http://<phone-ip>:8080/video`, DroidCam:
  `http://<phone-ip>:4747/video`), put the URL in `source` with `type: url`. Phone and PC must be on the same Wi-Fi.
  Replace the placeholder IP `192.168.1.50`.
- Feeds always keep only the newest frame, stamp arrival time right after capture, and reconnect automatically;
  a camera shows `failed` after 10 s down but keeps retrying.

**Glass-to-glass latency:** run `latency-timer`, point the camera at that screen, run `capture --show-timer` on the same
PC and take a screenshot: latency = live counter minus the counter visible in the camera image (repeat, use the median).

## Calibration

1. `calib board`: print `board.png` at 100 % on A4 landscape, mount it flat. Measure one square with a ruler and set
   `board.square_length_m` in [`configs/calibration.yaml`](configs/calibration.yaml).
2. Each camera: `uv run python -m livefire.app calib intrinsics --cam 0` (then `--cam 1`). Move the board over the whole
   image with varied tilt; captures happen automatically when the board is still (`s` forces one, `f` finishes, at
   least 20 views). Check the held-out RMS (target below 1 px).
3. Lay the board flat on the table where all cameras see it: `uv run python -m livefire.app calib extrinsics --cams 0,1`
   (averages 15 frames per camera). World origin = board centre, +Z up, table plane z = 0.
4. The result is `data/calib/calib_vNNN.yaml` and becomes active. Changing camera resolution requires recalibration.

Camera placement: 60-90 degrees apart, 1-1.5 m from the table, both seeing the table.

## Interactive viewer (click to ignite)

Mouse: left click ignites at the clicked surface (empty background and non-flammable objects only show a message),
right click selects an object. Keys: `1`-`8` set the selected object's material (paper, cloth, wood, plastic, metal,
glass, ceramic, other_nonflammable), `space` pause, `+`/`-` ignition radius, `o` geometry overlay, `v` single/dual view,
`r` re-acquire, `s` screenshot, `q` quit. `--out DIR` sets where screenshots go; `--record` saves a replayable session.
Today this runs on the synthetic scene; on real cameras it arrives with live mode (P1-006).

## Configuration

| File | Contents |
| --- | --- |
| `configs/app.yaml` | seed, device (`cpu` / `cuda:0`), sim time step |
| `configs/cameras.yaml` | live camera sources |
| `configs/calibration.yaml` | ChArUco board and calibration settings |
| `configs/depth.yaml` | depth model and table-plane anchoring |
| `configs/acquisition.yaml` | acquisition window, TSDF fusion, object clustering, default materials |
| `configs/grid.yaml` | region of interest and gas grid resolution |
| `configs/sim.yaml`, `configs/render.yaml` | simulation and rendering parameters (each with unit, range, status) |
| `configs/materials.yaml` | material table (PRD 8.5) |
| `configs/camera.yaml` | the synthetic two-camera rig |

## Project layout

```text
livefire/
  configs/            all tunables (YAML)
  src/livefire/
    common/           geometry contract, config, timers, interfaces, benchmark CSV
    capture/          camera sources (USB, IP, file, synthetic), recorder
    calib/            ChArUco board, intrinsics, extrinsics, calibration store
    perception/depth/ depth backends (Depth Anything V2), table-plane anchoring
    fusion/           depth accumulation, TSDF, surface extraction, objects
    scene/            synthetic scene, snapshot builder, surface elements, occlusion
    sim/              surface fire model, MAC gas solver, coupled simulation (Warp)
    render/           volume renderer, overlays, compositor
    app/              CLI, runner, interactive viewer, replay, HUD, calibration CLI
  tests/              pytest suite (headless)
  bench/              benchmarks, results CSVs, plots
  docs/               PRD, status, handover, architecture, ADRs, task files, phase reports
```

## Known limitations

- No fire spread between separate objects yet (radiative heat transfer is P2); fire stays on what you ignite.
- Flames look like a glowing column; the flame appearance model is P4.
- Single-camera depth accuracy on real scenes is unproven (the CPU spike on a public dataset gave 10-30 cm error);
  a two-view refinement step may be needed.
- Objects are found by geometry until segmentation (P3): a thin cloth lying on the table merges into the table.
- All benchmark numbers are CPU numbers (`gpu=none`); see `docs/PERFORMANCE_BUDGET.md`.

## Documentation

[`ARCHITECTURE`](docs/ARCHITECTURE.md) · [`NUMERICAL_METHODS`](docs/NUMERICAL_METHODS.md) ·
[`COORDINATE_CONVENTIONS`](docs/COORDINATE_CONVENTIONS.md) · [`TEST_STRATEGY`](docs/TEST_STRATEGY.md) ·
[`PERFORMANCE_BUDGET`](docs/PERFORMANCE_BUDGET.md) · [`decisions/`](docs/decisions) (ADRs) ·
[`tasks/`](docs/tasks) (plans) · [`reports/`](docs/reports) (phase reports) · [`literature`](docs/literature.md)

## Live mode runbook (burn a real object)

1. `uv run python -m livefire.app calib board --out board.png --dpi 300`; print at 100 %, measure one square, set `square_length_m` in `configs/calibration.yaml`.
2. Edit `configs/cameras.yaml` (USB index, or the phone URL such as `http://<phone-ip>:8080/video`).
3. Per camera: `calib intrinsics --cam N`. Then lay the board flat on the table in view of all cameras: `calib extrinsics`.
4. Put objects on the table (board may stay, keep objects off its footprint), leave the scene still, run `make live`.
   It captures about 5 s, builds the scene (the first run downloads the depth model), then opens the window.
5. Left click an object to ignite it. Right click selects, keys 1-8 set the material (default is paper), `o` shows the
   reconstructed geometry, `r` re-acquires after you move something, `q` quits.
