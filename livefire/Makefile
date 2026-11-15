.PHONY: live interactive capture capture-headless latency-timer setup test test-slow lint run run-rt bench bench-spike segment-spike plots calibrate eval eval-acquire measure eval-placement eval-alignment eval-mae eval-latency eval-m9 eval-ablation-uniform bench-radiation

setup:
	uv sync --all-groups --all-extras

test:
	uv run pytest -m "not slow" -q

test-slow:
	uv run pytest -q

lint:
	uv run ruff check src tests bench eval

# CPU-friendly offline demo; override e.g. `make run RUN_ARGS='--grid 64 --offline --seconds 10 --ignite ...'`
RUN_ARGS ?= --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.08"

run:
	uv run python -m livefire.app run --scene table_v1 --out out/demo $(RUN_ARGS)

# Real-time run (default grid from configs/grid.yaml), 20 wall seconds: for the GPU machine
run-rt:
	uv run python -m livefire.app run --scene table_v1 --out out/demo_rt --seconds 20 --ignite "-0.075,0.0,0.1@0.3:0.08"

bench:
	uv run python bench/run_bench.py

bench-radiation:
	uv run python bench/p2_radiation_compare.py

bench-spike:
	uv run python bench/spike/run_spike.py

# Segmentation model spike (P3-001, ADR-0007; needs `uv sync --all-extras`): writes bench/results/segment_spike_<cpu|gpu>_<date>.csv
segment-spike:
	uv run python bench/p3_segment_spike.py $(SEGMENT_ARGS)

plots:
	uv run python bench/plots.py

# e.g. make calibrate CALIB_ARGS="intrinsics --cam 0"   or   CALIB_ARGS="extrinsics --cams 0,1"
CALIB_ARGS ?= board --out board.png --dpi 300
calibrate:
	uv run python -m livefire.app calib $(CALIB_ARGS)

# Gate 1 evaluation tools (P1-007, docs/RUNBOOK_P1.md). `make eval` = no-hardware self-check of every tool on synthetic data
# (writes eval/results/*synthetic*.csv). On your rig: eval-acquire once, then the others with EVAL_ARGS="--bundle <acquisition.npz>".
eval:
	uv run python eval/placement.py --synthetic
	uv run python eval/alignment.py --selftest
	uv run python eval/geometry_mae.py --synthetic

# Acquire the real scene once (cameras + active calibration) and save out/eval/acq_<time>/acquisition.npz
eval-acquire:
	uv run python -m livefire.app measure --acquire-only $(EVAL_ARGS)

# M4: click pairs of points, type the tape distance (cm); e.g. make measure EVAL_ARGS="--bundle out/eval/acq_x/acquisition.npz"
measure:
	uv run python -m livefire.app measure $(EVAL_ARGS)

eval-placement:
	uv run python eval/placement.py $(EVAL_ARGS)

eval-alignment:
	uv run python eval/alignment.py $(EVAL_ARGS)

# e.g. make eval-mae EVAL_ARGS="--pairs out/eval/measure_x/pairs.csv --label desk1"
eval-mae:
	uv run python eval/geometry_mae.py $(EVAL_ARGS)

# e.g. make eval-latency EVAL_ARGS="out/live/live_x/latency.csv --label rtx4060"
eval-latency:
	uv run python eval/latency.py $(EVAL_ARGS)

# Live mode: active calibration + cameras in configs/cameras.yaml -> acquire -> simulate on live frames, click to ignite.
# e.g. make live LIVE_ARGS="--depth da2_relative_small --grid 48"
live:
	uv run python -m livefire.app live $(LIVE_ARGS)

# M9 spread monotonicity (P2-003): scripted + 12 dev + 12 held-out randomized scenarios x 4 target materials; CPU, long
# (see README); exit code 1 if M9 fails. `make eval-m9 M9_ARGS=--quick` is a smoke run.
M9_ARGS ?=
eval-m9:
	uv run python eval/p2_m9_spread.py $(M9_ARGS)

# Uniform vs material-dependent combustion and the demo-spread evidence (P2-003); CPU, long
ABL_ARGS ?=
eval-ablation-uniform:
	uv run python eval/p2_ablation_uniform.py $(ABL_ARGS)

# Interactive viewer on the synthetic scene: left click ignites, right click selects, keys 1-8 set material, q quits
interactive:
	uv run python -m livefire.app interactive --scene table_v1

# Live cameras (configs/cameras.yaml): window with both feeds, fps/age/skew overlays; q or Esc quits
capture:
	uv run python -m livefire.app capture

capture-headless:
	uv run python -m livefire.app capture --headless --seconds 5

# Glass-to-glass latency: film this window with a camera, then run `capture --show-timer`
latency-timer:
	uv run python -m livefire.app latency-timer
