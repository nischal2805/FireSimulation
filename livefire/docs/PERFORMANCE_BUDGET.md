# Performance budget

Spec: `docs/PRD.md` section 5.4 (memory and VRAM budget), section 5.5 (latency definitions), section 11 (metrics,
targets, benchmark rules in 11.4, CSV schema in Appendix C). This file records where the measured numbers live and
what is still missing. It contains no numbers itself.

## Measured results (CPU only)

All committed results are CPU runs in this container, labelled `gpu=none, framework=<fw>-cpu`. They are not GPU
timings and do not decide ADR-0001.

- Baseline: `bench/results/baseline_cpu_2026-11-14.csv`
- Spike (framework comparison kernels): `bench/results/spike_cpu_2026-11-14.csv`

Read the numbers from those files. Plots are generated from the CSVs only (`make plots`). Schema: PRD Appendix C,
implemented by `src/livefire/common/bench_csv.py`. Workloads are defined in `bench/workloads.yaml`.

## VRAM table

| Backend | Workload | Peak allocated (MB) | Peak reserved (MB) | Source |
| --- | --- | --- | --- | --- |
| pending GPU run | | | | |

The table stays empty until the GPU run on the RTX 4060 (`make bench`, `make bench-spike`). Until then,
`peak_vram_*` columns in the CSVs are empty for CPU runs.

## Rules

- Benchmark method follows PRD 11.4: fixed recorded inputs, warm-up discarded, at least 5 repeats, median and p95,
  power mode and GPU clocks recorded on the GPU machine.
- Do not copy CSV values into reports or docs. Point to the CSV path.
- Any mock or placeholder timing is marked and labelled in the output.
