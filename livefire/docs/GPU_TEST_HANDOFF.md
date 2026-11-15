# GPU test handoff (for a fresh chat on the RTX 4060 / Windows 11 machine)

Everything below was built and checked **headless on CPU only** in a cloud container (no GPU, no camera, no display).
Nothing here has run on your GPU, cameras or Windows yet. This document tells a new Claude Code chat (or you) what
exists, in what state, and exactly what to run and report back. Paste the "Prompt for the new chat" section into the
new chat first.

All fire is virtual. Nothing here controls real heat or flame.

## 1. Where the code is

- Repo `nischal2805/FireSimulation`, project in the `livefire/` subfolder. Branch **`main`** (same commit as
  `agent/magical-clarke-xah78d`) holds everything: P0, P1, P2-001/002, P3-001/002 and unverified work in progress.
- The old branches `agent/inspiring-maxwell-5od2va` (strict ancestor, older) and `worktree-agent-af7a64019b66f7883`
  (P2 work, already merged into main) are now redundant. They were not deleted.
- Read in order: `docs/PRD.md` (spec), `CLAUDE.md` (rules), `docs/STATUS.md`, `docs/reports/phase-1.md`,
  `docs/RUNBOOK_P1.md`, `README.md`.

## 2. Verified state versus unverified state (be honest in any report)

| Area | State |
| --- | --- |
| P0, P1-001..007 (live mode, click-to-ignite, calibration, acquisition, eval tools) | Implemented and reviewed. Fast suite last fully run at **305 passed, 2 skipped, 1 xfailed** (CPU) before P1-007 fixes. |
| P2-001/002 radiation (modes 0/1/2, default 2), ADR-0006 | Implemented and reviewed. Fast suite passed (274 passed, 4 deselected, 1 xfailed) at commit b7c7dbd on the P2 branch. |
| P3-001 segmentation backends, ADR-0007 | Implemented and reviewed, but review found a **bug in the spike script** (top prediction dropped), so the committed spike CSVs and ADR-0007 scores were invalid. The fix was in progress: the spike CSVs were deleted and **must be re-run** (`make segment-spike`). ADR-0007 numbers are not trustworthy until then. |
| P3-002 cross-view association | Implemented and reviewed (approved), plus a follow-up fix round that was **committed untested** (commit 92730bb "WIP"). |
| P2-003 scenarios, M9 eval, demo-spread tuning | **Unfinished and unreviewed** (commit 09b316c "WIP"). `m9_spread_2026-11-15.csv` and `demo_spread_2026-11-15.csv` are **partial** results from an interrupted run: do not quote them. `T_rad_flame` was changed 1100 -> 1400 in `configs/sim.yaml` for the demo spread (artistic, ADR-0006 addendum) and has not been reviewed. |
| P2-004..006, P3-003..006 (materials backends, review UI, modes/VRAM, M8 eval) | **Not started.** |
| Merge of the P2 branch into main | Conflicts in Makefile, README, `configs/sim.yaml` resolved by hand; only an import check was run afterwards, **no tests**. |
| Gates 1, 2, 3 | **None passed.** Gate 1 needs your real-camera runs. Gate 2 needs P2-003..006. Gate 3 needs P3-003..006. |

Known deviations: low-profile cloth merges into the table under pure geometric clustering (strict-xfail test in
`tests/test_acquire.py`; segmentation-based acquisition separates it); M2 latency gate uses the median (provisional);
a failed re-acquire ends in ERROR; placement thresholds and change-monitor values are untuned.

## 3. First steps on the GPU machine (in order)

```powershell
git clone https://github.com/nischal2805/FireSimulation.git
cd FireSimulation
git checkout main
cd livefire
uv sync --all-groups --all-extras
```

1. In `configs/app.yaml` set `device: "cuda:0"`.
2. Fast suite on CPU-defaults first, to separate merge problems from GPU problems:
   ```powershell
   uv run pytest -m "not slow" -q
   ```
   Expect roughly the counts above. **Any failure here is a real finding**: the merge and the WIP commits were never
   tested. Fix root causes (do not weaken tests) or report them.
3. Same suite on the GPU device (after step 1): `uv run pytest -m "not slow" -q`. Report device-specific failures
   (Warp CUDA, determinism, the golden-digest test is CPU-only and should skip).
4. Lint: `uv run ruff check src tests bench eval`.

## 4. GPU benchmarks and spikes (all write CSVs; commit them)

| Command | Purpose | Notes |
| --- | --- | --- |
| `uv run python bench/spike/run_spike.py` (`make bench-spike`) | ADR-0001: Warp vs Taichi on GPU | decides ADR-0001 (still "proposed") |
| `uv run python bench/run_bench.py` (`make bench`) | M1/M2b baseline on the 4060 | CSV in `bench/results/` with `gpu`, `driver`, `cuda` filled |
| `uv run python bench/p2_radiation_compare.py` (`make bench-radiation`) | radiation cost per mode on the GPU | CPU numbers exist; GPU is the missing column |
| `uv run python bench/depth_spike.py` | ADR-0005 depth backends on real data/GPU | needs `--extra depth` torch cu128 |
| `uv run python bench/p3_segment_spike.py` (`make segment-spike`) | **re-run required** (invalid CSVs were removed) | downloads Grounding DINO tiny + SAM 2.1 tiny + Mask2Former; reports GPU latency and VRAM |
| `uv run python eval/p2_m9_spread.py` (`make eval-m9`) | M9 monotonicity (unfinished script) | use `--quick` first; the full run took far too long on CPU |
| `uv run python eval/p2_ablation_uniform.py` (`make eval-ablation-uniform`) | uniform vs material-dependent ablation | unfinished |

Record power mode, clocks, nothing else running (PRD 11.4). Update ADR-0001 and ADR-0005 from the measured data;
re-score ADR-0007 from the re-run spike.

## 5. Real-camera workflow (Gate 1)

Follow `docs/RUNBOOK_P1.md` end to end (print the ChArUco board, measure a square, place two cameras 60-90 degrees
apart 1-1.5 m from the table, calibrate intrinsics per camera and extrinsics, then `uv run python -m livefire.app live`).
Send back: `eval/results/*.csv` from alignment, geometry MAE, placement and latency; screenshots with the `o` overlay;
the `out/live/live_<time>/latency.csv`. Things the first real run must check: monocular depth accuracy (biggest
risk, `docs/HANDOVER.md` section 4), that the overlay lines up, change-monitor false triggers, HUD latency numbers.

Offline demo (no cameras): `uv run python -m livefire.app run --scene table_v1 --out out/demo_rt --seconds 20 --ignite "-0.075,0.0,0.1@0.3:0.08"`
and `uv run python -m livefire.app interactive --scene table_v1` (click to ignite).

## 6. Work still to do (continue in the new chat; plans exist)

- `docs/tasks/P2-plan.md`: finish **P2-003** (review/finish the WIP: scenarios, M9, ablation, demo spread), then
  P2-004 (missing PRD 8.6 invariants, solver quality metrics), P2-005 (resolution sensitivity G=64 vs 128 and the
  10-minute long run: these are fast on the GPU), P2-006 (NUMERICAL_METHODS + phase-2 report).
- `docs/tasks/P3-plan.md`: P3-001 fix round (re-run spike, fix ADR numbers; review items listed in
  the ledger below), P3-002 minor items (some committed untested), then P3-003 materials, P3-004 review UI and
  overrides, P3-005 acquisition/simulation modes + VRAM table, P3-006 M8 eval + phase-3 report.
- Open review items not yet fixed: P3-001 `review1` list (spike metric bug, p95/p99 aggregation, COCO hashes over https,
  category canonicalisation, box cap/mask-IoU dedup before SAM 2, missing-extra error types, `release()` semantics);
  P3-002 item 3 (guard against a table mask leaking onto objects, deferred to P3-005/006) and items 6, 8.
- The development method used: one implementer subagent per task, then spec + quality review, fix rounds, then a
  whole-branch review before a gate. Keep it, and keep running the test suite before every commit.

## 7. Prompt for the new chat (paste this)

> I'm continuing the LiveFire Previs project on my Windows 11 + RTX 4060 machine. Repo `nischal2805/FireSimulation`,
> branch `main`, project folder `livefire/`. Read `livefire/docs/GPU_TEST_HANDOFF.md`, then `docs/PRD.md`,
> `CLAUDE.md`, `docs/STATUS.md`. Do section 3 first (setup, set `device: "cuda:0"`, run the fast test suite on CPU and on
> GPU, fix or report every failure with root causes, never weaken tests). Then run the section 4 benchmarks and commit
> the CSVs, update ADR-0001/0005/0007 from measured data, and report what failed. Then continue with section 6. Do not
> quote any number that a committed script did not produce, and state plainly which gates are not passed.
