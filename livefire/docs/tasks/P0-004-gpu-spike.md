## Task 4: P0-004 GPU framework spike (Warp vs Taichi) and ADR-0001

**Files:** `bench/spike/{common.py,warp_kernels.py,taichi_kernels.py,numpy_ref.py,run_spike.py}`,
`tests/test_spike_kernels.py`, `docs/decisions/ADR-0001-gpu-framework.md`, `bench/results/spike_cpu_<date>.csv`.

- Two kernels, identical semantics in NumPy reference, Warp, Taichi:
  1. **Advection:** semi-Lagrangian advection of a scalar on a cell-centred G^3 grid by a cell-centred velocity,
     RK2 backtrace, trilinear interpolation with clamp-to-edge, `dt`, `h` given.
  2. **Conduction gather:** surface-element graph update `T_new_i = T_i + dt * sum_j G_ij (T_j - T_i) / Cap_i` with
     `neighbors i32 (N,K)` -1 padded and `G (N,K)` f32.
- Correctness tests: Warp and Taichi each match the NumPy reference (advection max abs err < 1e-4 on G=32 seeded
  random field; conduction < 1e-5 on N=5000, K=16). Taichi tests are skipped if taichi is not importable.
- `run_spike.py --device cpu|cuda:0 --out <csv>`: timings for advection G in {64, 96, 128} and conduction
  N in {20k, 100k}; warm-up 5 launches discarded (includes JIT), 5 repeats x 20 samples, median and p95, with proper
  device sync for timing. Writes Appendix C CSV rows (columns verbatim from PRD Appendix C; `gpu` = `none` on CPU;
  `framework` = `warp-<ver>`/`taichi-<ver>`; `timing_method` = `perf_counter_ns+sync`). Records JIT/compile time
  separately in `notes`. `make bench-spike` runs it with `device` from app.yaml.
- Also report: lines of kernel code, interop with NumPy/PyTorch (zero-copy?), CPU fallback quality, determinism of
  repeated runs, maintenance status and Blackwell (sm_120, RTX 5050) support claimed in each project's release notes
  (cite the URL you read; if you cannot verify, write "unverified").
- ADR-0001 (template PRD Appendix B) with status **proposed** (GPU timings pending the user's 4060 run), scored by PRD
  11.6 with the CPU evidence, recommendation and fallback. Implementer writes the evidence and a draft score; the
  controller finalises the decision.

**Acceptance:** correctness tests pass for both frameworks; CPU spike CSV committed; ADR drafted.

