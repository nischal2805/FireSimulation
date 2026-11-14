# ADR-0001: GPU framework for simulation and renderer kernels (Warp vs Taichi)

Date: 2026-11-14. Status: **proposed** (draft by implementer; GPU timings pending the user's RTX 4060 run; the
controller finalises the decision).

## Context

PRD 1.2/Global constraints: no hand-written CUDA C++; kernels must also run on a CPU device because the
development container has no GPU (`configs/app.yaml: device` = `cpu` here, `cuda:0` on the user's machine). Phase 0
needs a Python GPU framework for the gas solver, surface conduction and the volume renderer (Task 5+). PRD 0.2
asks for at most 3 candidates and a half-day spike; two candidates are clearly dominant for this use, so two were
spiked.

## Candidates (source, date, license, size)

| | NVIDIA Warp | Taichi |
| --- | --- | --- |
| Install size (Linux x86_64 wheel, PyPI JSON) | 149.3 MB (`manylinux_2_28_x86_64`, includes CUDA libs) | 56.2 MB (`cp311 manylinux_2_27_x86_64`) |
| Version tested | warp-lang 1.19.0 (PyPI upload 2026-11-02) | taichi 1.7.4 (PyPI upload 2025-07-31) |
| License | Apache-2.0 (PyPI metadata) | Apache-2.0 (PyPI metadata) |
| Backends | CPU (LLVM), CUDA (NVRTC) | CPU (LLVM 15.0.4, bundled), CUDA, others |
| Source | https://pypi.org/pypi/warp-lang/json ; https://raw.githubusercontent.com/NVIDIA/warp/main/CHANGELOG.md | https://pypi.org/pypi/taichi/json |

## Method (data, metrics, hardware, commands)

- Two kernels with one NumPy reference (`bench/spike/numpy_ref.py`), identical semantics in Warp
  (`warp_kernels.py`) and Taichi (`taichi_kernels.py`): (1) semi-Lagrangian advection of a scalar on a
  cell-centred G^3 grid, RK2 midpoint backtrace, trilinear interpolation with clamp-to-edge; (2) conduction
  gather `T_new_i = T_i + dt * sum_j G_ij (T_j - T_i) / Cap_i`, `neighbors i32 (N,K)` -1 padded, `G (N,K)` f32.
- Correctness: `tests/test_spike_kernels.py` (seeded; advection G=32, conduction N=5000,K=16).
- Timing: `bench/spike/run_spike.py --device cpu|cuda:0 --out <csv>`; advection G in {64,96,128}, conduction N in
  {20k,100k} K=16 (synthetic random graph, neighbours within +-256 indices, `# MOCK`); 5 warm-up launches
  discarded (the first includes JIT, reported in `notes`, kernel caches disabled/redirected so JIT is cold),
  5 repeats x 20 samples = 100 samples, median/p95/p99, device sync before and after each sample
  (`perf_counter_ns+sync`). Output follows PRD Appendix C columns.
- Hardware here: container, 4 CPU cores, no GPU, Linux 6.18, Python 3.11.18. **Only `--device cpu` was run.**
  The `cuda:0` code path exists (Warp device string; Taichi `ti.cuda`) but is **untested**.
- Commands: `uv sync --all-groups --extra spike`; `uv run pytest tests/test_spike_kernels.py`;
  `uv run python bench/spike/run_spike.py --device cpu`.

## Results (table, CSV links)

CSV: `bench/results/spike_cpu_2026-11-14.csv`. CPU only, `gpu=none`; framework labelled `<fw>-<ver>-cpu`.

| Kernel | size | Warp cpu median / p95 ms | Taichi cpu median / p95 ms |
| --- | --- | --- | --- |
| advection | G=64 | 10.116 / 15.160 | 4.004 / 5.839 |
| advection | G=96 | 34.543 / 39.388 | 11.818 / 17.610 |
| advection | G=128 | 82.697 / 91.862 | 23.802 / 28.476 |
| conduction | N=20k | 0.857 / 0.892 | 0.334 / 0.453 |
| conduction | N=100k | 4.076 / 5.885 | 1.377 / 1.912 |

Caveat: Warp's CPU backend ran single-threaded (process CPU time / wall = 0.99), Taichi used about 3 of 4 cores
(2.9). The ratio is therefore mostly thread count, not code-generation quality, and says little about GPU speed.

Other evidence (details in `.superpowers/sdd/P0-plan/task-4-report.md`):

- Correctness: both match the NumPy reference (advection < 1e-4 including a large-displacement clamping case;
  conduction < 1e-5 on a [0,50) K field and < 5e-7 relative on 293-793 K). Repeat runs are bitwise identical in both.
- Kernel code: 59 non-blank, non-comment lines each (5 decorated functions each); near line-for-line equivalent.
- JIT (cold, CPU, from the CSV `notes`): Warp about 1.9 s for the module (both kernels); Taichi about 93 ms
  (advection) + 47 ms (conduction). Only rows labelled "cold JIT first launch" in the CSV are true cold compiles.
- Interop: Warp CPU `.numpy()` is a zero-copy view, `wp.array(..., copy=False)` aliases a NumPy buffer
  (default copies), and `wp.to_torch/from_torch/to_dlpack/from_dlpack/to_jax` exist (PyTorch not installed
  here, not exercised). Taichi `ti.types.ndarray` kernel arguments accept external NumPy (and, per Taichi docs,
  torch) arrays directly: verified for NumPy (`test_taichi_kernel_accepts_numpy_directly`, error < 1e-4). Taichi
  `ndarray.from_numpy/to_numpy` do copy. The spike runner keeps device-resident `ti.ndarray`s so timings exclude
  host transfers. Net: interop is comparable; Warp has more explicit torch/DLPack helpers.
- Maintenance: Warp 1.19.0 on 2026-11-02, releases 1.17.0 (2026-08-31), 1.18.0 (2026-10-05), 1.19.0rc1
  (2026-10-23), actively developed by NVIDIA. Taichi last release 1.7.4 on 2025-07-31 (about 15 months before
  today), previous 1.7.3 on 2024-12-23.
- Blackwell (sm_120, RTX 50): evidence is equal and unverified for both (no release note found claiming
  sm_120 for either). Warp CHANGELOG (read in full text) has no explicit sm_120 entry; README states PyPI
  wheels use CUDA Toolkit 13.4, need driver R580+ and a Turing (sm_75) or newer GPU, which includes sm_120, and
  kernels are compiled at runtime by NVRTC. A web-search snippet (search-snippet only) reported a Warp 1.17/1.18
  issue reproduced on an RTX PRO 6000 Blackwell (sm_120). Taichi: no release note or issue found
  (unverified); the installed build bundles LLVM 15.0.4 (observed), and whether it can target sm_120 is untested.
  This only matters for the user's RTX 4060 (Ada, sm_89) if the project later moves to an RTX 50 machine.

## Scores (11.6)

Score 0-5 per criterion, weight in brackets; total = sum(score x weight) / 5. Draft by implementer.

| Criterion | Warp | Taichi | Basis |
| --- | --- | --- | --- |
| Accuracy on project data (35) | 5 | 5 | Both match NumPy reference within thresholds |
| Latency and VRAM on RTX 4060 (30) | 3 | 3 | **Provisional**: GPU not measured; CPU favours Taichi (3x, mostly threads) so no change |
| Robustness (20) | 4 | 3 | Blackwell evidence equal (unverified for both), so scored evenly. Margin from release cadence/maintenance (Warp 2026-11 vs Taichi 2025-07), NVIDIA backing, CUDA 13 wheels. Warp CPU is single-threaded |
| Implementation risk (10) | 4 | 4 | Same kernel size; both accept NumPy; Warp has torch/DLPack helpers, Taichi has ~20x faster cold JIT; both need Python-subset kernel syntax |
| License (5) | 5 | 5 | Both Apache-2.0 |
| **Total / 100** | **82** | **78** | |

Hard limits: none triggered (no budget or VRAM figure exists yet; licences permit use).

## Decision and fallback

Controller decision (recorded here; scores above are the implementer's draft): **Warp is the primary GPU
framework; Taichi is the config-selectable fallback**, with the spike kernels kept in `bench/spike/` as the
porting base. Status stays **proposed** until `make bench-spike` with device `cuda:0` on the RTX 4060 confirms it.
The fallback triggers if Warp fails on the 4060 or 5050, or loses by more than 1.5x on the GPU spike.

## Consequences

- Task 5+ kernels are written in Warp, using `device` from `configs/app.yaml`; CPU tests run single-threaded, so
  CPU test grids stay at G=32 (slow marker for longer runs). Cold JIT on first launch (about 2 s here for two
  kernels) needs a warm-up step and Warp's kernel cache.
- Simulation and render backends own their device arrays; use `.numpy()` views for host readback.
- A Taichi fallback would add `taichi` (optional `spike` group) as a runtime dependency; kernels port from
  `bench/spike/taichi_kernels.py`.
- Status stays proposed until the GPU run exists; CUDA path is untested in this container.
