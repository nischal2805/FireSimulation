.PHONY: setup test test-slow lint run bench bench-spike plots calibrate eval

setup:
	uv sync --all-groups

test:
	uv run pytest -m "not slow" -q

test-slow:
	uv run pytest -q

lint:
	uv run ruff check src tests bench

run:
	uv run python -m livefire.app run --scene table_v1 --seconds 8 --ignite "-0.075,0.0,0.05@0.5" --out out/demo

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
