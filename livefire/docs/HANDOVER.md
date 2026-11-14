# Handover: LiveFire Previs (for the next agent)

Written 2026-11-14 at the end of a session. Read this, then `docs/PRD.md` (authority), `CLAUDE.md` (rules),
`docs/STATUS.md`, `docs/tasks/P1-plan.md`. Everything is on branch `agent/inspiring-maxwell-5od2va` of
`nischal2805/FireSimulation`; the project lives in the `livefire/` subfolder.

## 1. Where the project is

| Phase | State |
| --- | --- |
| P0 contracts + synthetic vertical slice | Done. Gate 0 approved by the user (CPU-only evidence, deviation R2). Report: `docs/reports/phase-0.md`. |
| P1 real-camera geometry, live capture, click-to-ignite | In progress. Tasks P1-001..P1-005 implemented and reviewed (see 3). P1-006 live mode, P1-007 Gate 1 tools, P1-008 report not started. |
| P2..P5 | Not started. |

User priority (stated explicitly): **live cameras (USB webcam + Android phone as IP camera) and clicking on the live
feed to start a fire**, running on their Windows 11 laptop with an RTX 4060 (CUDA and uv installed).

## 2. Environment facts that shape every decision

- The development container used so far had **no GPU, no camera, no display**. Every module was built to be tested
  headless (synthetic renders, recorded clips, local MJPEG servers). Nothing has run on the user's GPU yet.
- The user's machine is **Windows 11** (cmd/PowerShell; `make` may be missing). README has a "Windows (no make)"
  table; keep it in sync with the Makefile. No bash, pathlib only.
- Warp CPU is single-threaded: CPU timings (`bench/results/*cpu*.csv`) say nothing about the 4060.
- The user asked for token-frugal work and no unrequested artifacts.

## 3. What exists (P1 so far)

| Task | Code | Notes |
| --- | --- | --- |
| P1-001 live sources | `capture/opencv_source.py`, `file_source.py`, `factory.py`, `record.py`, `app/capture_cli.py`, `configs/cameras.yaml` | USB (DSHOW on Windows, `fourcc: MJPG`), URL (phone "IP Webcam" `http://<ip>:8080/video`), file; newest-frame-wins, t_arrive stamped right after grab, reconnect forever, "failed" = down > 10 s. `python -m livefire.app capture`, `latency-timer`. |
| P1-002 click-to-ignite | `app/click.py`, `app/interactive.py` (pure controller), `app/window.py` (thin OpenCV loop) | `python -m livefire.app interactive --scene table_v1` works on the synthetic scene. Click depth = centre pixel of the frozen snapshot `view_depth`; never ignites a different object than the one hit (tested vs renderer ground truth); ambiguous edges return "no surface". Keys: left click ignite, right click select, 1-8 material, space, +/-, o, v, r, s, q. |
| P1-003 calibration | `calib/*`, `app/calib_cli.py`, `configs/calibration.yaml` | ChArUco 7x5, 35 mm squares (user must measure the print), held-out RMS, extrinsics averaged over 15 frames, world = board centre on table, +Z up. `python -m livefire.app calib board|intrinsics|extrinsics`. |
| P1-004 depth | `perception/depth/*`, `bench/depth_spike.py`, ADR-0005 (proposed) | Depth Anything V2 Small (relative) primary, metric-indoor fallback; per-view anchoring to the calibrated table plane (P1R3). torch is the optional `depth` extra (CPU wheels on Linux, cu128 on Windows). |
| P1-005 acquisition | `fusion/*`, `scene/acquire.py`, `configs/acquisition.yaml` | Accumulation, Warp TSDF, surfel extraction, table-plane + clustering objects, snapshot. See its report section in git log / commit message for numbers. |

All P0 machinery (sim, MAC gas solver, flame field, renderer, compositor, replay, benchmarks) is described in
`docs/ARCHITECTURE.md`, `docs/NUMERICAL_METHODS.md` and ADR-0001..0004.

## 4. Open risks (highest first)

1. **Monocular depth accuracy.** CPU spike on TUM RGB-D: anchored MAE 0.095-0.32 m, strict anchor rejected most
   frames. Gate 1 needs M6 median < 3 cm. Expect to add a two-view refinement step (triangulated matches, or a
   wide-baseline multi-view model such as MASt3R-class) once real clips show the error. Do this as a time-boxed ADR.
2. **Nothing verified on the RTX 4060 / Windows**: Warp CUDA path, deterministic atomics
   (`wp.config.deterministic = RUN_TO_RUN` set in `sim/device.py`), torch cu128, DSHOW cameras, OpenCV window,
   tkinter screen fit. ADR-0001 (Warp vs Taichi) is still "proposed" until `bench-spike` runs on the GPU.
3. **No fire spread between objects** (P0 has conduction + convection only). Radiation options are P2 (PRD 8.2).
   On real scenes the user will see fire stay on the clicked object.
4. **Occlusion tolerance** `view_depth` override uses ~9 mm (tuned on exact synthetic depth). With noisy real depth
   it may push occlusion around; retune with real data (ruling in section 6).
5. Literature review (`docs/literature.md`) is search-snippet only (container blocked full-text fetch).

## 5. Where to steer next (recommended order)

1. **P1-006 live mode** (`docs/tasks/P1-plan.md`, Task 6): `python -m livefire.app live` = calibrate-load ->
   acquire -> READY -> simulate on live frames with the P1-002 controller, geometry overlay (`o`), change monitor,
   camera drop/recover, latency HUD. This is the feature the user is waiting for. Test headless with synthetic clips
   + `OracleDepthBackend`, then hand the user a runbook.
2. **Get real data early**: give the user a short runbook (print board, measure square, calibrate each camera,
   extrinsics, record 10 s clips with `capture --record`), and run `make test`, `bench-spike`, `bench`,
   `depth_spike` on their GPU. Use the results to finalise ADR-0001/0005 and decide on depth refinement (risk 1).
3. **P1-007** Gate 1 tools (alignment M6, measured-distance M4, placement study, latency) and `docs/RUNBOOK_P1.md`.
4. **P1-008** phase report; ask the user for Gate 1 approval. Then P2 (radiation for spread, invariants,
   resolution sensitivity), P3 (segmentation + material recognition to replace geometric clustering + default
   `paper`), P4 (flame look, relight, performance on GPU), P5 (evaluation, demo).

Pending small chores: Makefile/README rows listed in the reports of P1-001 (fourcc note, status meanings), P1-004
(`depth-spike`, `uv sync --extra depth`) and P1-005 were not all applied; apply them with the Windows table.

## 6. Decisions already made (do not silently undo; use an ADR to change)

P0: Warp primary/Taichi fallback (ADR-0001, proposed); MAC grid (ADR-0002); energy split + re-radiation + clamps
(ADR-0003); combustion-intensity field `flame` with `flame_ref` 3e4 W/m^3 (ADR-0004); edge-flux conduction;
snapshot host-side read-only, device mirrors owned by backends; soot in kg/m^3; hole policy (splat, else
known-clear background, else hole); `ignite()` sets burning directly; `make run` offline G=48 by default, `run-rt`
real time.

P1 (`docs/tasks/P1-plan.md` "Controller rulings" plus these): one or two cameras; world frame from the board;
depth anchored to the table plane; geometric-clustering objects with default material `paper` flagged for review;
clicks use the frozen snapshot `view_depth`, never live depth; a camera is "failed" after 10 s down but keeps
retrying; click depth = centre pixel, ambiguous edges -> "no surface"; material overrides live in the controller +
replay log (snapshot stays frozen); `view_depth = min(splat, measured)` where measured is nearer by more than
max(0.5 spacing, 0.5 % depth) plus a fringe rule at silhouettes; ADR-0005 depth choice proposed.

## 7. How the work was run (reuse if useful)

Subagent-driven development: one implementer per task (task brief extracted from the plan), then a spec+quality
review, fix rounds with scoped re-reviews, then a whole-branch review before each gate. Every task copies its brief
into `docs/tasks/`, commits with a co-author trailer, and never edits `docs/STATUS.md` (controller-owned). Reviews
caught real bugs every time (silhouette click errors, renderer crash off-view, capture timestamps after decode), so
keep the review step. Numbers in reports only from committed CSVs; CPU numbers labelled CPU.

## 8. Physical steps only the user can do

Print `python -m livefire.app calib board` at 100 %, measure a square, set `square_length_m`; place two cameras
60-90 deg apart, 1-1.5 m from the table; run intrinsics per camera and extrinsics with the board flat on the table;
record clips; run the GPU commands in the README; later, tape-measure distances for M4.
