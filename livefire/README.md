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
| P1 | Real cameras: live capture, click-to-ignite, ChArUco calibration, depth, fusion, live mode | In progress: capture, click-to-ignite, calibration, depth, acquisition and live mode (`live`) are built; **live mode has only been tested on synthetic clips, never on real cameras or the GPU** |
| P2-P5 | Simulation quality, scene understanding, visual fidelity and performance, final evaluation | Not started |

Everything so far was developed and tested on a CPU-only Linux machine. Nothing has run on a GPU or on Windows yet;
treat the GPU and Windows paths as untested.

## What you can run today

| Feature | Command |
| --- | --- |
| **Live mode: your cameras, click the feed to start a virtual fire** (needs calibration first, see "Live mode") | `uv run python -m livefire.app live` |
| Click on the synthetic scene to start fires | `uv run python -m livefire.app interactive --scene table_v1` |
| Offline two-view fire demo video | `uv run python -m livefire.app run --scene table_v1 --out out/demo --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.08"` |
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

`--all-extras` installs the optional `spike` (Taichi), `depth` (PyTorch + Transformers) and `perception` (segmentation and materials: PyTorch, torchvision, Transformers) extras. PyTorch comes from
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
| `make lint` | `uv run ruff check src tests bench eval` | lint |
| `make run` | `uv run python -m livefire.app run --scene table_v1 --out out/demo --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.08"` | offline demo video `out/demo/composite.mp4` |
| `make run-rt` | `uv run python -m livefire.app run --scene table_v1 --out out/demo_rt --seconds 20 --ignite "-0.075,0.0,0.1@0.3:0.08"` | real-time run (needs a GPU to be real time) |
| `make live` | `uv run python -m livefire.app live` | live mode on your cameras (extra options: `make live LIVE_ARGS="--depth da2_relative_small --grid 48"`) |
| `make interactive` | `uv run python -m livefire.app interactive --scene table_v1` | click-to-ignite viewer |
| `make capture` | `uv run python -m livefire.app capture` | live camera check |
| `make capture-headless` | `uv run python -m livefire.app capture --headless --seconds 5` | camera check without a window |
| `make latency-timer` | `uv run python -m livefire.app latency-timer` | on-screen ms counter for glass-to-glass latency |
| `make calibrate` | `uv run python -m livefire.app calib board --out board.png --dpi 300` | print-ready board (other `calib` subcommands below) |
| `make eval` | `uv run python eval/placement.py --synthetic` then `uv run python eval/alignment.py --selftest` then `uv run python eval/geometry_mae.py --synthetic` | no-hardware self-check of the Gate 1 tools on synthetic data (CSVs in `eval/results/`) |
| `make eval-acquire` | `uv run python -m livefire.app measure --acquire-only` | acquire your real scene once, save `out/eval/acq_<time>/acquisition.npz` |
| `make measure` | `uv run python -m livefire.app measure --bundle out/eval/acq_<time>/acquisition.npz` | M4: click two points, type the tape distance (cm), saves `pairs.csv` |
| `make eval-placement` | `uv run python eval/placement.py --bundle out/eval/acq_<time>/acquisition.npz` | camera placement study (angle, shared area, depth coverage); exit 1 with guidance if poor |
| `make eval-alignment` | `uv run python eval/alignment.py --bundle out/eval/acq_<time>/acquisition.npz` | M6: board corners reconstructed vs known |
| `make eval-mae` | `uv run python eval/geometry_mae.py --pairs out/eval/measure_<time>/pairs.csv` | M4: MAE in cm from the measured pairs |
| `make eval-latency` | `uv run python eval/latency.py out/live/live_<time>/latency.csv` | M2: latency statistics from a live run |
| `make bench` | `uv run python bench/run_bench.py` | benchmarks to `bench/results/*.csv` |
| `make bench-radiation` | `uv run python bench/p2_radiation_compare.py` | radiation modes 0/1/2: cost CSV `bench/results/radiation_cpu_<date>.csv` and spread CSV `eval/results/radiation_effect_<date>.csv` (ADR-0006; about 33 min on CPU, `--quick` for a smoke run) |
| `make eval-m9` | `uv run python eval/p2_m9_spread.py` | M9 spread monotonicity (PRD 11.1): scripted, dev and held-out scenarios, CSVs `eval/results/m9_spread_<date>.csv` and `m9_summary_<date>.csv`; exit 1 if M9 fails (CPU; see the CSV `wall_s`; `--quick` for a smoke run) |
| `make eval-ablation-uniform` | `uv run python eval/p2_ablation_uniform.py` | uniform vs material-dependent combustion (PRD 11.2) `eval/results/ablation_uniform_<date>.csv` and the demo-spread evidence `demo_spread_<date>.csv` (CPU, about 30 min; `--part ablation|demo`, `--quick`) |
| `make bench-spike` | `uv run python bench/spike/run_spike.py` | Warp vs Taichi kernel spike (decides ADR-0001) |
| `make plots` | `uv run python bench/plots.py` | plots from the CSVs |
| - | `uv run python bench/depth_spike.py` | depth model spike (ADR-0005) |
| `make segment-spike` | `uv run python bench/p3_segment_spike.py` | segmentation model spike (ADR-0007; needs `uv sync --extra perception` or `--all-extras`; optional `--fetch-coco`) |
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

Camera placement: 60-90 degrees apart, 1-1.5 m from the table, both seeing the table. `eval/placement.py` checks it.

## Gate 1 measurements

The step-by-step procedure (print the board, place the cameras, calibrate, run live, measure M2/M4/M6, which files to send
back) is in [`docs/RUNBOOK_P1.md`](docs/RUNBOOK_P1.md). The tools are `eval/placement.py`, `eval/alignment.py`,
`eval/geometry_mae.py`, `eval/latency.py` and `python -m livefire.app measure`; results go to `eval/results/`.
Pass a region that starts with a minus sign as `--region=-0.6,-0.4,0.6,0.4`.

## Interactive viewer (click to ignite)

Mouse: left click ignites at the clicked surface (empty background and non-flammable objects only show a message),
right click selects an object. Keys: `1`-`8` set the selected object's material (paper, cloth, wood, plastic, metal,
glass, ceramic, other_nonflammable), `space` pause, `+`/`-` ignition radius, `o` geometry overlay, `v` single/dual view,
`r` re-acquire, `s` screenshot, `q` quit. `--out DIR` sets where screenshots go; `--record` saves a replayable session.
This viewer runs on the synthetic scene; live mode (below) uses the same controller on your cameras.

## Live mode (your cameras)

All fire is virtual: the simulation runs on a frozen 3D snapshot of your table and is composited over the live camera
frames. Real objects are never heated or ignited.

**What you do, once:**

1. Print the board, measure a square, set `square_length_m`, calibrate each camera, then run `calib extrinsics` with the
   board flat on the table (see Calibration). Keep camera resolution unchanged afterwards.
2. List your cameras in `configs/cameras.yaml` (USB index, or the phone's `http://<ip>:8080/video`). The cam_ids must match
   the calibration. Check them with `uv run python -m livefire.app capture`.
3. Install the depth model (needs internet once, then cached): `uv sync --all-groups --all-extras`. Metric depth
   (`da2_metric_small`) is the default; `--depth da2_relative_small` is the alternative. Set `device: "cuda:0"` in
   `configs/app.yaml` on the RTX 4060.

**Each session:**

1. Put objects on the table (paper, wood, metal ...) inside the area both cameras see. Remove the board or leave it flat.
   Do not touch the scene or the cameras from here on.
2. `uv run python -m livefire.app live` (Windows without make) or `make live`.
3. The window walks through CALIBRATING (checks the stored calibration against the board if it is visible; a warning in the
   HUD means "recalibrate"), ACQUIRING (about 5 s collecting frames, then depth and fusion; progress is shown), READY,
   then SIMULATING.
4. Left click on an object in either view to start a fire there. Objects default to **paper** (flagged REQUIRES REVIEW):
   right click an object, press `1`-`8` to set its real material (1 paper, 2 cloth, 3 wood, 4 plastic, 5 metal, 6 glass,
   7 ceramic, 8 non-flammable); metal and glass will not ignite. `+`/`-` change the ignition radius, `space` pauses,
   `v` toggles single/dual view, `s` saves a screenshot, `q` or Esc quits.
5. **`o` shows the geometry overlay**: the reconstructed surface drawn on each live view (colour = material). It should hug
   the real objects; if it floats or is offset, recalibrate or re-check camera placement before trusting the fire.
6. If a person or object moves in the static scene for about a third of a second the HUD shows
   **"scene changed: press r to re-acquire"** and the simulation pauses. `r` re-acquires (fire resets); `space`
   continues on the old snapshot. Re-acquiring resets the fire **and your material tags and selection** (tags that survive re-acquisition arrive with P3 segmentation). If one camera delivers no frames at start-up, live mode runs in single view (HUD: DEGRADED) and `r` re-acquires with it once it works. If a camera is down for 3 s it is dropped and the HUD says DEGRADED (single view); it
   comes back when it reconnects.
7. The HUD shows capture-to-display latency, frame age, simulation lag and fps. Each run writes `out/live/live_<time>/`
   with `latency.csv` (one row per camera per frame), `events.jsonl` (clicks, material changes), `summary.json` and
   screenshots. Glass-to-glass latency is measured separately with `latency-timer` (see Capture).

Limits today: fire spreads only within the clicked object (no radiation between objects yet); geometry accuracy from
single-image depth is unverified on real data; all thresholds in `configs/live.yaml` are untuned defaults; a live session
cannot be replayed with `replay` (the snapshot comes from your cameras and is not stored).

## Configuration

| File | Contents |
| --- | --- |
| `configs/app.yaml` | seed, device (`cpu` / `cuda:0`), sim time step |
| `configs/cameras.yaml` | live camera sources |
| `configs/calibration.yaml` | ChArUco board and calibration settings |
| `configs/live.yaml` | live mode: calibration check, change monitor, camera drop (untuned defaults) |
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
