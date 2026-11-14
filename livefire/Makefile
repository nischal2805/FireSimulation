.PHONY: capture capture-headless latency-timer setup test test-slow lint run run-rt bench bench-spike plots calibrate eval

setup:
	uv sync --all-groups --all-extras

test:
	uv run pytest -m "not slow" -q

test-slow:
	uv run pytest -q

lint:
	uv run ruff check src tests bench

# CPU-friendly offline demo; override e.g. `make run RUN_ARGS='--grid 64 --offline --seconds 10 --ignite ...'`
RUN_ARGS ?= --grid 48 --offline --seconds 6 --ignite "-0.075,0.0,0.1@0.3:0.06"

run:
	uv run python -m livefire.app run --scene table_v1 --out out/demo $(RUN_ARGS)

# Real-time run (default grid from configs/grid.yaml), 20 wall seconds: for the GPU machine
run-rt:
	uv run python -m livefire.app run --scene table_v1 --out out/demo_rt --seconds 20 --ignite "-0.075,0.0,0.1@0.3:0.06"

bench:
	uv run python bench/run_bench.py

bench-spike:
	uv run python bench/spike/run_spike.py

plots:
	uv run python bench/plots.py

calibrate:
	@echo "not implemented until P1"; exit 2

eval:
	@echo "not implemented until P1"; exit 2

# Live cameras (configs/cameras.yaml): window with both feeds, fps/age/skew overlays; q or Esc quits
capture:
	uv run python -m livefire.app capture

capture-headless:
	uv run python -m livefire.app capture --headless --seconds 5

# Glass-to-glass latency: film this window with a camera, then run `capture --show-timer`
latency-timer:
	uv run python -m livefire.app latency-timer
