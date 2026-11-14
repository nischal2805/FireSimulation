## Task 11: P0-011 Benchmark harness and baseline CSV

**Files:** `bench/run_bench.py`, `bench/workloads.yaml`, `bench/plots.py`, `src/livefire/common/bench_csv.py`
(`BenchmarkRecorder` implementation), `tests/test_bench_csv.py`, `bench/results/baseline_cpu_<date>.csv`,
`bench/plots/*.png`.

- `BenchmarkRecorder` writes rows with exactly the PRD Appendix C columns, in order; fills environment fields
  automatically (git sha, date, OS, python, framework version, gpu/driver/cuda via `nvidia-smi` if present else
  `none`, power_mode `unknown` unless provided). VRAM fields from framework counters when on GPU, empty on CPU.
- Workloads (`workload_version: 1`): `sim_step` coupled at G in {64, 96, 128} on table_v1 with paper burning
  (pre-roll 120 steps after ignition so fire exists), `surface_step`, `gas_step`, `projection`, `render_view`
  (one view at 720p, volume_scale 0.5 and 1.0), `frame_two_views` (sim run_frame + 2 renders + composite).
  Rules PRD 11.4: warm-up discarded, >= 5 repeats, median and p95 (and p99), `samples` = total timed samples.
  `--quick` flag for CI-size runs. `make bench` runs the full set with the device from app.yaml.
- `bench/plots.py` reads only CSVs and writes scaling plots (stage ms vs G) to `bench/plots/`; `make plots`.
- Test: schema/column order, a row round-trip, required fields non-empty.
- Run the baseline here (CPU) and commit the CSV and plots, labelled CPU. If the full G=128 set exceeds 30 minutes
  on CPU, reduce samples (not the workload) and record that in `notes`.

**Acceptance:** CSV committed with all workloads; plots generated from it.

