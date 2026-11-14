# LiveFire Previs

Live two-camera virtual-fire previs. All fire is virtual. See `docs/PRD.md` for the spec and `docs/STATUS.md` for progress.

P0 is a synthetic vertical slice (no AI, no cameras).

Repository: https://github.com/nischal2805/FireSimulation (this project lives in the `livefire/` subfolder of that repo).

## Setup

Requires Python 3.11 exactly and [uv](https://docs.astral.sh/uv/). The environment is locked by `uv.lock`.

```
make setup   # uv sync --all-groups --all-extras (creates .venv, Python 3.11; --all-extras keeps taichi for the spike)
```

Run tools through `uv run` (the make targets do this for you).

## Commands

| Target | What it does |
| --- | --- |
| `make test` | fast tests (`pytest -m "not slow"`) |
| `make test-slow` | all tests, including the long-run and resolution tests |
| `make lint` | `ruff check src tests bench` |
| `make run` | offline CPU-friendly demo on the synthetic `table_v1` scene (writes to `out/demo`); change with `RUN_ARGS=...` (default `--grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.06"`) |
| `make run-rt` | real-time run, default grid from `configs/grid.yaml`, 20 s wall, same ignition (writes to `out/demo_rt`) |
| `make bench` | benchmark harness, writes CSVs to `bench/results/` |
| `make bench-spike` | framework spike kernels benchmark |
| `make plots` | plots generated from the CSVs in `bench/results/` |
| `make calibrate CALIB_ARGS="..."` | calibration tool (see Calibration below), default prints the board |
| `make interactive` | interactive viewer on the synthetic scene (click to ignite) |
| `make eval` | placeholder, not implemented until P1 |

## Windows (no make)

`make` is optional; on Windows 11 run the commands directly from the project root (PowerShell or cmd). No Python code
path shells out to bash; paths use `pathlib`.

| Make target | Equivalent |
| --- | --- |
| `make setup` | `uv sync --all-groups --all-extras` |
| `make test` | `uv run pytest -m "not slow" -q` |
| `make test-slow` | `uv run pytest -q` |
| `make lint` | `uv run ruff check src tests bench` |
| `make run` | `uv run python -m livefire.app run --scene table_v1 --out out/demo --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.06"` |
| `make run-rt` | `uv run python -m livefire.app run --scene table_v1 --out out/demo_rt --seconds 20 --ignite "-0.075,0.0,0.1@0.3:0.06"` |
| `make bench` | `uv run python bench/run_bench.py` |
| `make bench-spike` | `uv run python bench/spike/run_spike.py` |
| `make plots` | `uv run python bench/plots.py` |
| `make capture` | `uv run python -m livefire.app capture` |
| `make capture-headless` | `uv run python -m livefire.app capture --headless --seconds 5` |
| `make latency-timer` | `uv run python -m livefire.app latency-timer` |
| `make interactive` | `uv run python -m livefire.app interactive --scene table_v1` |
| `make calibrate` | `uv run python -m livefire.app calib board --out board.png --dpi 300` (print-ready board, A4 landscape, print at 100 %) |
| | `uv run python -m livefire.app calib intrinsics --cam 0` (and `--cam 1`; `--from-dir DIR` for saved images): per-camera intrinsics |
| | `uv run python -m livefire.app calib extrinsics --cams 0,1` (`--from-dir DIR` with cam0.png, cam1.png): world poses, writes calib_vNNN |

Replay a recorded run: `uv run python -m livefire.app replay out/demo` (on a GPU device the final state is compared
with a tolerance, see ADR-0002).

## Live cameras and latency (P1)

Cameras are listed in `configs/cameras.yaml` (1 or 2 entries; `type: usb|url|file|synthetic`). A USB camera is an
index (DirectShow on Windows by default). A phone works as an IP camera: Android "IP Webcam" serves
`http://<phone-ip>:8080/video`, DroidCam `http://<ip>:4747/video`; put the URL in `source` (phone and PC on the same
Wi-Fi). Edit the placeholder `192.168.1.50` first.

`python -m livefire.app capture [--record DIR] [--seconds N] [--headless] [--show-timer]` shows both feeds with
per-camera measured fps, frame age (ms since arrival), inter-camera skew and connection status
(`connected/reconnecting/failed`). `--record DIR` writes `cam<N>.mp4`, `timestamps.csv` and `manifest.yaml` (sha256).
Press `q` or Esc to quit.

Glass-to-glass latency (PRD 5.5), per camera:
1. Run `python -m livefire.app latency-timer` and leave its big millisecond counter on a monitor.
2. Point the camera at that monitor and run `python -m livefire.app capture --show-timer` (on the same PC; both
   counters use the wall clock modulo 100 000 ms).
3. Take a screenshot: the camera feed shows an old counter value, the "live timer" panel shows the current one.
   Glass-to-glass latency = live value minus the value seen in the feed (repeat a few times; report the median;
   it includes monitor refresh and capture latency, so it is an upper bound, resolution about one monitor frame).

## Calibration (P1)

1. `calib board`: print `board.png` at 100 % scale on A4 landscape, mount flat. MEASURE one square with a ruler; set `board.square_length_m` in `configs/calibration.yaml` (marker length scales automatically).
2. Per camera: `calib intrinsics --cam N`. Move the board through the whole image with varied tilt; auto-capture fires when the board is still and the pose is new; the heat-map shows uncovered regions; press `s` to force a capture, `f` to finish (>= 20 views). Read the HELD-OUT RMS line (target < 1 px).
3. Lay the board flat on the table, visible to all cameras, not moving. `calib extrinsics --cams 0,1`, SPACE per camera. World origin = board centre, +Z up toward the cameras; table plane z = 0. Check the reported reprojection RMS and consistency (mm).
4. Result is `data/calib/calib_vNNN.yaml`, set active. Changing camera resolution requires recalibration (mismatch raises).

## Interactive viewer (P1)

`python -m livefire.app interactive --scene table_v1` opens the two composited views (synthetic scene, so you can
try it before cameras are calibrated). Mouse: left click ignites at the clicked surface point (holes and
non-flammable objects only show a message), right click selects an object. Keys: `1`..`8` set the selected object's
material (paper, cloth, wood, plastic, metal, glass, ceramic, other_nonflammable), `space` pause/resume, `+`/`-`
ignition radius, `o` geometry overlay, `v` single/dual view, `r` re-acquire request, `s` screenshot, `q` quit. Screenshots
go to `--out DIR` (default `out/interactive`); `--record` also records the session there (events.jsonl with every click
and override, replayable with `replay DIR`). The window is sized to the screen (`--max-width/--max-height` override).

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
