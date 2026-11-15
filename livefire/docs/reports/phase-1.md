# Phase 1 report: real-camera geometry, live mode and Gate 1 tools

Date: 2026-11-15. Branch `agent/magical-clarke-xah78d`. Spec: `docs/PRD.md` (section 12 "P1"). Plan: `docs/tasks/P1-plan.md`.
Every number below is quoted from a committed CSV/JSON or from a test count recorded in a task report; CPU numbers are
labelled CPU, synthetic outputs are labelled synthetic.

## Gate 1 status

**NOT YET EVALUATED: pending the user's physical runs.** Gate 1 is **not passed**. The development container has no
camera, no GPU and no display, so no real-scene M4, M6 or latency number exists. The code and the evaluation tools are
built and tested on synthetic data with known answers only. Outcomes follow PRD 0.4 (`PASSED`,
`PASSED_WITH_APPROVED_DEVIATION`, `FAILED`); none can be chosen before the runs below.

All commands run from `livefire/` on the Windows 11 machine (PowerShell is fine); full steps are in `docs/RUNBOOK_P1.md`.
`<BUNDLE>` is the path printed by the acquire step (`out/eval/acq_<time>/acquisition.npz`).

| Gate 1 criterion | What was built | Command(s) the user runs | Send back |
| --- | --- | --- | --- |
| M6 fusion alignment on a real scene, both cameras (median under 3 cm) | `eval/alignment.py`, shared acquisition bundle, ChArUco corner detection or `--landmarks` CSV | `uv run python -m livefire.app measure --acquire-only`, then `uv run python eval/alignment.py --bundle <BUNDLE> --label desk_scene1`; elevated check: add `--landmarks landmarks.csv --label desk_scene1_elevated` | `eval/results/alignment_*.csv` (summary and `_landmarks`), `out/eval/acq_*/` (npz, `view_cam*.png`, `depth_cam*.png`) |
| M4 geometry accuracy against tape-measured distances | `measure` click tool, `eval/geometry_mae.py` | `uv run python -m livefire.app measure --bundle <BUNDLE>` (10+ pairs, 20 better), then `uv run python eval/geometry_mae.py --pairs out/eval/measure_<time>/pairs.csv --label desk_scene1` | `eval/results/geometry_mae_desk_scene1.csv`, `out/eval/measure_*/pairs.csv` |
| Fire runs on real geometry with manual materials | `live` mode: acquire, READY, simulate on the frozen snapshot, click-to-ignite, material keys `1`-`8`, geometry overlay `o` | `uv run python -m livefire.app live` (right click + `1`-`8` to tag, left click to ignite, `o` overlay, `s` screenshots) | screenshots with overlay on and a fire, `out/live/live_*/` (`summary.json`, `events.jsonl`) |
| Baseline performance on the RTX 4060 (M2 latency; M1 via `make bench`) | `LatencyRecorder`, `eval/latency.py`, glass-to-glass tools (`latency-timer`, `capture --show-timer`) | `live` for 60 s or more with a fire and both cameras, then `uv run python eval/latency.py out/live/live_<time>/latency.csv --label rtx4060_dual --glass "0=..." --glass "1=..."`; also `make test-slow`, `make bench-spike`, `make bench` (set `device: "cuda:0"` in `configs/app.yaml`) | `eval/results/latency_rtx4060_dual.csv`, `out/live/live_*/latency.csv`, new `bench/results/*.csv`, `nvidia-smi` output |
| Original and achieved targets recorded separately | PRD originals are unchanged; the tools report measured values next to the PRD targets (M6 3 cm, M2 100 ms) | nothing extra; achieved values come from the CSVs above | none |

Camera placement (PRD 7.1) is part of the same session: `uv run python eval/placement.py --bundle <BUNDLE>`; send
`eval/results/placement_*.csv`. Also send `data/calib/calib_vNNN.yaml` with its manifest, `configs/cameras.yaml`,
`configs/calibration.yaml`, and a photo of the rig. Calibration itself (board print, `calib intrinsics` per camera,
`calib extrinsics`) comes first; see the runbook sections 1 to 6.

## Tasks

| Task | What exists | Commits | Review outcome |
| --- | --- | --- | --- |
| P1-001 | USB/URL/file camera sources, newest-frame-wins capture with arrival stamp, reconnect, `capture` and `latency-timer` tools, recording | 03f867e, 6a06054 | reviewed, one fix round |
| P1-002 | Click-to-ignite controller and OpenCV viewer on the synthetic scene (`interactive`) | 057c407, ab53e44, da76add | reviewed, two fix rounds (silhouette-safe click depth, lifecycle gating) |
| P1-003 | ChArUco calibration (intrinsics with held-out RMS, extrinsics, world = board centre), `calib` CLI | 67d1baa, ff3ae16 | reviewed, fix batch |
| P1-004 | Depth backends (Depth Anything V2 small, metric and relative), table-plane anchoring, depth spike, ADR-0005 | 73f3989, ff3ae16 | reviewed, fix batch; ADR-0005 stays proposed |
| P1-005 | Accumulation, Warp TSDF fusion, surfel extraction, table plane + geometric-clustering objects, snapshot, acquisition | 13425f0, 35710af | review 1 needed fixes, clean after one round |
| P1-006 | `live` mode: CALIBRATING, ACQUIRING, READY, SIMULATING on live frames, overlay, change monitor, camera drop/recover, latency CSV | da5c1f5, 9953648 | review 1 needed fixes, clean after one round |
| P1-007 | `eval/{placement,alignment,geometry_mae,latency}.py`, `measure` tool, `configs/placement.yaml`, `docs/RUNBOOK_P1.md` | 5839f70, f325dd0, f00fc65, f41946b | review 1 needed fixes, clean after one round |
| P1-008 | this report | (this commit) | not reviewed |

Task briefs are in `docs/tasks/P1-00*.md`. P2-001 (radiation) landed in 83e88d2 and e00d88a; it is outside Gate 1.

## Headless evidence (no hardware)

Last full fast-suite result recorded: **305 passed, 2 skipped, 1 xfailed (CPU, `uv run pytest -m "not slow" -q`)**, run
during P1-007 before its fix commits (f325dd0, f00fc65, f41946b). Those fixes changed wording, messages, a
`ValueError` and test MOCK marking; ruff and `tests/test_eval_p1.py` (26 passed) were re-run afterwards. The full suite
was not re-run on the final P1-007 commit. The one xfail is the strict cloth test below.

Synthetic outputs that exist (all CPU, oracle depth, scene `table_v1`; none is a Gate 1 result):

| File | Content (synthetic) |
| --- | --- |
| `eval/p1_005_acquisition.json` | acquisition with oracle depth plus noise (MOCK): 7926 elements, table-plane deviation 1.46 mm, tilt 0.024 deg, geometry p95 2.93 mm, acquisition 9.35 s total (CPU, includes a 4.87 s capture window) |
| `eval/results/geometry_mae_synthetic_oracle.csv` | MAE 0.214 cm (cam0) and 0.213 cm (cam1), AbsRel 0.0016 and 0.0017, completeness 0.996 and 1.000, ROI pixels only |
| `eval/results/placement_synthetic_table_v1_oracle_depth_cpu.csv` | camera angle 66.75 deg, shared fraction 0.666, valid coverage 1.0, visible fraction 0.75 and 0.74; all thresholds pass |
| `eval/results/alignment_selftest.csv` | tool self-test with an injected 0.02 m depth offset: median 2.01 cm; rows are marked "SYNTHETIC SELFTEST ... not a Gate 1 result" and `meets_m6` is blank |

No latency CSV is committed, so no latency number is quoted anywhere. Depth-model numbers exist only from the P0/P1-004
CPU spike (`bench/results/depth_spike_cpu_2026-11-14.csv`, TUM RGB-D, 10 frames): with strict anchoring
`da2_relative_small` was accepted on 2/10 frames (MAE 0.095 m) and `da2_metric_small` on 0/10; with the lenient
diagnostic anchor 9/10 (MAE 0.255 m) and 8/10 (MAE 0.323 m). The anchor plane there was the dominant ground-truth plane,
which favours anchoring. CPU latency 525 to 593 ms median at 640x480. These say nothing about the RTX 4060.

## Deviations and rulings

| # | Item | Detail | Needs the user |
| --- | --- | --- | --- |
| a | Low-profile cloth merges into the table | Geometric clustering cannot separate a 3 mm cloth from the table at about 1 % depth noise. Recorded as strict xfail `test_cloth_separated_from_table` in `tests/test_acquire.py`. Resolved by P3 segmentation. Controller ruling: accepted P1 deviation. | no |
| b | M2 pass rule | Gate on the median `t_present - t_arrive` <= 100 ms; p95, p99 and max are reported alongside, never gated. The PRD says "100 ms or less" without a percentile; the M1/M2 wording uses medians. Provisional controller ruling. | **confirm** |
| c | Failed re-acquire ends in ERROR | The lifecycle table (`ALLOWED[ACQUIRING] = {READY, ERROR}`) is a PRD contract and was not edited; the window exits with a message and must be restarted. Revisit via ADR if it happens in practice. | only if it bites |
| d | Camera absent at start-up | Live mode starts in single view (HUD: DEGRADED). The missing camera is not added automatically (the compositor needs its frozen view); the HUD says "press r to re-acquire" when it appears. | no |
| e | M6 independence | Depth is anchored to the table plane over the board footprint, so board corners on the table are only weakly independent of the reconstruction. Elevated landmarks (`--landmarks`, for example tape marks on a box top) are recommended; the CSV notes and runbook say so. | recommended |
| f | Untuned thresholds | Placement thresholds in `configs/placement.yaml` (shared 0.5, valid coverage 0.8, visible 0.3, visibility tolerance 3 cm) and `configs/live.yaml` change-monitor values (threshold 0.03, 10 frames, drop 3 s, verify 2 px) are untuned defaults. | tune on the rig |
| g | ADRs | ADR-0001 (Warp vs Taichi) and ADR-0005 (depth backend) remain **proposed** until run on the GPU. | after GPU runs |

Other small rulings: the commit trailer session URL is the current session, not the one named in the plan text; the
P1-007 latency tool skips the first 30 frames and non-simulating states by default (overridable); an unmarked synthetic
landmarks CSV was removed by the controller (f41946b).

## Open risks

1. **Monocular depth accuracy (highest).** Gate 1 needs M6 median under 3 cm; the only real-image evidence is the CPU
   TUM spike above (cm-level on the anchor plane, 0.1 to 0.3 m MAE elsewhere, strict anchor rejects most frames). If the
   real M6 or M4 is poor, the next step is a time-boxed ADR for two-view refinement (triangulated matches or a
   wide-baseline multi-view model) or the Depth Anything 3 candidate that was not run.
2. **Nothing verified on the RTX 4060 or Windows:** Warp CUDA path, deterministic atomics, torch cu128 wheels, DSHOW
   cameras, the OpenCV window and mouse mapping, tkinter screen fit, PowerShell quoting in the runbook.
3. **Real-time budget unknown:** no GPU acquisition or live timing exists; M2 and M3 (15 s to READY) are unmeasured.
4. **View-depth occlusion tolerance** (about 9 mm) was tuned on exact synthetic depth and may misbehave with noisy real depth.
5. **Change monitor** may false-trigger under real lighting; a trigger pauses the simulation.
6. Literature review is search-snippet only (P0).

## What the first real run should check

1. Calibration quality: held-out RMS below 1 px per camera; square size measured across several squares.
2. `capture`: both feeds near 30 fps with frame age well under 100 ms; resolutions match the calibration.
3. `placement.py` passes; if it fails with an overlay that looks right, retune `visible_depth_tol_m` first.
4. Live mode: reaches READY and SIMULATING; press `o` and confirm the overlay hugs the real objects in both cameras.
   An offset or floating overlay means a calibration or depth problem; do not trust the fire results then.
5. Do M6 on board corners and again with elevated landmarks; compare the two (deviation e).
6. Click-to-ignite lands on the clicked object; metal and glass do not ignite; the change monitor does not fire without a real change.
7. Try both `--depth da2_metric_small` (default) and `--depth da2_relative_small`; the spike favoured relative on the anchor plane.

## Known limitations

- Fire spreads only within the clicked object in the P1 baseline; radiation between objects is P2 (P2-001 is committed but not part of Gate 1).
- Objects come from geometric clustering with default material `paper` flagged for review; real recognition is P3.
- Re-acquire resets the fire, material tags and selection; tags that survive arrive with P3.
- A live session is not replayable (the snapshot is not stored); events are logged only.
- Zero-thickness objects such as paper can show edge ghosts in fused geometry (synthetic observation, handover note).
- M4 edge accuracy is not implemented; AbsRel and completeness exist only for synthetic ground truth.
- Placement is not wired into `live`; the `measure` window loop and `run_window` could not be exercised without a display.
- `measure` needs both cameras at the same resolution.

## Next

1. The user runs the Gate 1 sequence above and sends the files; we record originals and achieved values separately, then
   propose the Gate 1 outcome (`PASSED`, `PASSED_WITH_APPROVED_DEVIATION` or `FAILED`) for approval.
2. P2 (simulation quality: radiative spread, invariants, resolution sensitivity) has started with P2-001.
3. P3 (automatic scene understanding: segmentation and material recognition, which also resolves deviation a) is in
   progress in the working tree and is not part of this report.
4. P4 (visual fidelity and GPU performance) and P5 (final evaluation, demo) follow.
