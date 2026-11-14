# Phase 0 report: contracts and synthetic vertical slice

Date: 2026-11-14. Branch `agent/inspiring-maxwell-5od2va`. Spec: `docs/PRD.md`. Plan: `docs/tasks/P0-plan.md`.
All numbers below are quoted from committed files; the source of each is named.

## Gate 0 outcome

**Not final yet. Recommended outcome: `PASSED_WITH_APPROVED_DEVIATION`, once the user's RTX 4060 run confirms the
two open items below.** The approved deviation is R2 (plan approval): this container has no GPU, so all evidence is
CPU-only (`gpu=none`).

| Gate 0 criterion (PRD 12) | Status | Evidence |
| --- | --- | --- |
| Synthetic two-camera scene renders the same shared fire state correctly in both views | Met (CPU) | `tests/test_render.py` cross-view centroid test (median error 2.44 px vs 6 px limit at 720p, task-9 report) and occlusion test with control; `tests/test_flame_field.py` asserts flame emission in both views from one coupled state; `make run` -> `out/demo/composite.mp4` (not committed, regenerable) |
| Geometry and invariant tests pass | Met on CPU, one item open | `make test`: 163 passed, 3 deselected (slow) at the final fix commit. The 10-min-simulated long-run invariant (slow) passed at commit 9587742 (G=32, 36000 steps, no NaN/Inf, bounds held; ADR-0003). It was **not re-run at HEAD** (stopped at the user's request; flame field and stage refactor landed since). Open item 1: `make test-slow` on the 4060. |
| Baseline benchmark CSV committed | Met (CPU) | `bench/results/baseline_cpu_2026-11-14.csv`, 18 rows, git_sha 4e9cc58, `gpu=none, warp-1.19.0-cpu`. Open item 2: GPU baseline via `make bench` on the 4060. |
| `make test` green | Met (CPU) | as above; GPU path changes (deterministic Warp mode, tolerance replay) unverified on CUDA |

Resolution-sensitivity invariant (PRD 8.6) is non-gating in P0 (ruling R9) and currently degenerate: the cloth never
ignites at G=32/64 because P0 has no radiative spread (`tests/test_coupled.py`, xfail).

## What was built (tasks)

P0-001 skeleton, config, timers; P0-002 geometry (PRD 4); P0-003 contracts and material table (PRD 9, 8.5);
P0-004 Warp vs Taichi spike, ADR-0001; P0-005 synthetic scene; P0-006 snapshot builder (surfels, obstacles,
per-view occlusion); P0-007 surface fire model; P0-008 MAC gas solver and coupled sim, ADR-0002/0003; P0-008b
combustion-intensity field, ADR-0004; P0-009 volume renderer and compositor; P0-010 app (lifecycle, runner, HUD,
replay, degradation ladder levels 1-2); P0-011 benchmark harness; P0-012 docs; P0-013 literature review.
Every task had a spec + quality review and fix rounds; a final whole-branch review found one crash (renderer
launch with an off-view active region) and GPU-path issues, all fixed in commits e767f0f, 557b0db, 63829e8.

## Benchmarks (CPU only, from `bench/results/baseline_cpu_2026-11-14.csv`)

| Workload | G | median ms | p95 ms |
| --- | --- | --- | --- |
| sim_step (coupled) | 64 | 655.8 | 711.0 |
| sim_step | 96 | 2288.6 | 2479.2 |
| sim_step | 128 | 5868.6 | 6253.1 |
| projection (RBGS 60 iters) | 96 | 1720.6 | 1867.1 |
| surface_step | 96 | 0.60 | 0.79 |
| render_view 720p, volume_scale 0.5 | 96 | 156.6 | 166.7 |
| frame_two_views | 96 | 4775.8 | 4930.2 |

Warp's CPU backend is single-threaded, so these numbers say nothing about the RTX 4060. The pressure projection
dominates the step. Spike CSV: `bench/results/spike_cpu_2026-11-14.csv`.

## ADRs

ADR-0001 GPU framework: Warp primary, Taichi fallback, **proposed** until `make bench-spike` on the 4060.
ADR-0002 MAC grid storage (public `vel` stays cell-centred). ADR-0003 energy split, surface re-radiation, clamps.
ADR-0004 combustion-intensity field `flame` (W/m^3) for visible flames.

## Deviations and rulings (full list in the session hand-off)

- CPU-only evidence (R2, approved). Surface elements from back-projected depth, TSDF deferred to P1 (R3).
- ADR-0004 added a public `FireState` field: with physical parameters the coarse-grid gas reaches only 376 K at G=64
  (probe measurement in the task-8b report), below the 700 K emission threshold; flames were invisible.
- `flame_ref` retuned 1e6 -> 3e4 W/m^3 (artistic, measured).
- `make run` defaults to offline G=48 for CPU; `make run-rt` is the real-time variant.

## Known limitations

- No fire spread between objects: P0 heat transfer is conduction + convection only; radiation options are P2.
- Flames read as a glowing column; the flame appearance model (world-space detail) is P4.
- Literature review is search-confirmed only (full-text fetch blocked by the container network).
- GPU determinism of float atomics relies on Warp's run-to-run deterministic mode: unverified on CUDA.
- Material overrides change the sim only, not the snapshot's object registry (P3).

## Reproduction

Linux/macOS: `make setup && make test && make run && make bench && make plots`. Windows without make: see the
README "Windows (no make)" table. For the GPU, set `device: "cuda:0"` in `configs/app.yaml`, then run
`make test-slow`, `make bench-spike`, `make bench`.

## Next (needs the user)

1. Run on the RTX 4060: `make test-slow`, `make bench-spike` (decides ADR-0001), `make bench`; commit the CSVs.
2. Approve the Gate 0 outcome. Then P1 (real cameras: phone/IP camera capture, ChArUco calibration, depth,
   click-to-ignite on the live feed).
