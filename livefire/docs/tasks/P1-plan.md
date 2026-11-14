# P1 implementation plan (real-camera geometry, live capture, click-to-ignite)

Spec: `docs/PRD.md` (authority; P1 = section 12 "P1: real-camera geometry", 7.1, 7.2, 5.1-5.6). Rules: `CLAUDE.md`.
Gate 0 approved by the user on 2026-11-14. User priority for P1: **live camera (USB webcam and phone over IP) and
click-on-the-feed-to-ignite**, working end to end on their Windows 11 + RTX 4060 machine.

## Global Constraints (bind every task)

- Everything in the P0 plan's Global Constraints still holds (`docs/tasks/P0-plan.md`, top section): Python 3.11 +
  uv, geometry contract (PRD 4) verbatim, YAML tunables in `configs/` with `{value, unit, range, status}` for
  physical/artistic parameters, seeded randomness, no fabricated numbers, mocks marked `# MOCK:`, Warp for kernels,
  device from `configs/app.yaml`, ruff clean, docstrings with shapes/units/frames/device/ownership.
- **Development container has no camera, no GPU, no display.** Every feature must be testable headless here with
  recorded clips, synthetic renders and local servers; the interactive window loop is a thin layer over pure,
  tested logic. The user runs live on Windows 11 with an RTX 4060: no bash, pathlib only, no POSIX-only calls.
  Add every new command to both the Makefile and the README "Windows (no make)" table.
- OpenCV: switch the dependency from `opencv-python-headless` to `opencv-python` (needs highgui for the window).
- PyTorch and model code are an optional extra `depth` (torch, transformers, accelerate if needed). `uv` sources:
  CPU wheel index on Linux, CUDA 12.8 wheel index on Windows. Core code must import without torch installed.
- Per task: copy the brief to `docs/tasks/<P1-ID>-<slug>.md`; commit `P1-00N: <title>` with a `Co-Authored-By`
  trailer naming your model and `Claude-Session: https://claude.ai/code/session_01FvsiJ1w9gRFcZuiYu1oxY3`;
  do not push; do not edit `docs/STATUS.md` (controller-owned).
- Measured numbers in reports must come from committed CSV/eval outputs. CPU numbers labelled CPU.

## Controller rulings (made at plan time)

- P1R1 One or two cameras: the system works with 1 or 2 configured cameras (single-view fallback, PRD 13); the
  thesis demo uses 2. Cameras are listed in `configs/cameras.yaml` (sources: USB index or URL).
- P1R2 World frame at calibration: the ChArUco board lies flat on the table; world origin = board centre, +X along
  the board's x axis (squares_x direction), +Z = board normal pointing up toward the cameras; the table plane is
  z = 0. The board may be removed after calibration; the world stays attached to the table.
- P1R3 Depth anchoring: monocular depth is anchored per view to the calibrated table plane (pixels whose ray hits
  z = 0 inside a configurable table-region polygon or the board footprint): fit scale (metric models) or scale+shift
  (relative models) robustly (RANSAC/Huber). This is the P1 "scale anchor" mitigation from PRD 14.
- P1R4 Objects before segmentation (P3): geometric clustering. Fit the table plane (RANSAC, must agree with z = 0
  within tolerance), remove it, connected components of remaining surfels (same 3D radius rule as neighbours) become
  objects; table = obj 0. Material defaults: table `other_nonflammable`; untagged objects take
  `acquisition.default_object_material` (default `paper`, flagged `requires_review=True`, shown in the HUD) so
  click-to-ignite works out of the box; the user re-tags with keys.
- P1R5 Click-to-ignite: a click at full-resolution pixel (u, v) of view c uses the frozen snapshot `view_depth[c]`
  (never live depth) to back-project to a world point and calls `sim.ignite(point, radius)`. Hole pixels and
  non-flammable targets do nothing except a HUD message. Clicks are recorded as replayable events
  `{t_sim, step, cam, u, v, point_world, radius}`.
- P1R6 Live frames are only composited; simulation runs on the frozen snapshot (PRD 1 "live" meaning). Perception
  models are released after READY (`torch.cuda.empty_cache` on CUDA).

---

## Task 1: P1-001 Live camera sources and capture tool

**Files:** `src/livefire/capture/opencv_source.py`, `src/livefire/capture/file_source.py`,
`src/livefire/capture/factory.py`, `src/livefire/capture/record.py`, `configs/cameras.yaml`,
`src/livefire/app/capture_cli.py` (+ wiring in `app/__main__.py`), `pyproject.toml` (opencv-python),
`tests/test_capture_live.py`, Makefile + README rows.

- `OpenCVCameraSource(CameraSource)`: source = int (USB index) or str (URL: MJPEG over HTTP such as the Android
  "IP Webcam" app `http://<phone-ip>:8080/video`, DroidCam `http://<ip>:4747/video`, or RTSP). Background thread
  grabs continuously (`grab()` loop, `retrieve()` only for the newest), keeps only the newest frame, stamps
  `t_arrive_ns = perf_counter_ns()` on arrival, increments `seq`, tracks measured fps and dropped/stale counts.
  Requested width/height/fps from config (best effort; report actual). On read failure: reconnect with exponential
  backoff (0.5 s .. 5 s), status `connected/reconnecting/failed` exposed. Windows: prefer `cv2.CAP_DSHOW` for USB
  indexes (config `backend: auto|dshow|msmf|v4l2|ffmpeg`). Frames are BGR from OpenCV: convert to the RGB `Frame`
  contract. `calib_version` attached from the active calibration (0 if uncalibrated).
- `VideoFileSource(CameraSource)`: plays a recorded clip as a camera, real-time paced (default) or as-fast-as-read,
  loop option; same newest-frame-wins semantics; used for tests, replay and benchmarks.
- `make_source(cfg) -> CameraSource` factory for `type: usb|url|file|synthetic`.
- `configs/cameras.yaml`: list of cameras `{cam_id, type, source, width, height, fps, backend}`; ship an example with
  cam0 = USB index 0 and cam1 = phone URL placeholder `http://192.168.1.50:8080/video`.
- Capture tool `python -m livefire.app capture [--record DIR --seconds N]`: opens a window with both feeds side by
  side, overlays per-camera measured fps, frame age (ms since t_arrive), inter-camera skew (ms between newest
  t_arrive values), connection status; `--headless` prints a summary instead. `--record` writes per-camera MP4 +
  `timestamps.csv` (seq, t_arrive_ns) + `manifest.yaml` (source, resolution, fps, sha256 of files, date).
- Glass-to-glass helper: `python -m livefire.app latency-timer` shows a large ms counter window to film with a
  camera; `capture --show-timer` overlays the live counter next to the camera feed so a screenshot shows both values
  (difference = glass-to-glass latency, PRD 5.5). Document the procedure in the README.
- Tests: VideoFileSource on a small generated MP4 (seq increases, newest-frame-wins, loop); OpenCVCameraSource
  against a local MJPEG HTTP server started in a thread (`http.server`, multipart/x-mixed-replace of generated JPEG
  frames with frame numbers encoded as pixel values): frames arrive, newest wins (consumer reading slower than
  producer sees skipped seq), reconnect after the server is stopped and restarted, status transitions; RGB order
  verified; recorder writes files + manifest with correct hashes.

**Acceptance:** tests pass headless; `capture --headless --seconds 2` on a file source prints fps/skew.

## Task 2: P1-002 Interactive viewer with click-to-ignite (synthetic first)

**Files:** `src/livefire/app/interactive.py` (pure controller logic), `src/livefire/app/window.py` (thin OpenCV
window loop), `src/livefire/app/click.py`, `src/livefire/app/__main__.py` (`interactive` command),
`tests/test_click_ignite.py`, `tests/test_interactive.py`, Makefile/README rows.

- `pixel_to_world(snapshot, cam_id, u, v) -> (point_world | None, reason)`: uses `snapshot.view_depth[cam_id]` and
  `view_hole_mask`; samples a small 3x3 neighbourhood median of valid depth to be robust at edges; returns None with
  reason `hole|outside|no_depth`. Also `pick_object(snapshot, cam_id, u, v) -> obj_id | -1` (nearest element to the
  back-projected point within 2 surfel spacings).
- `InteractiveController`: pure state machine fed with events (`click(cam,u,v,button,mods)`, `key(k)`, `tick(wall_dt)`)
  and producing actions on the sim / renderer and HUD messages: left click = ignite (P1R5); right click = select object
  (highlight); keys `1..8` = set selected object's material (order of MaterialTable, recorded as override event);
  `space` pause/resume; `+`/`-` ignition radius (bounded by sim.yaml range); `o` toggle geometry overlay; `v` toggle
  single/dual view; `r` request re-acquire (lifecycle PAUSED -> ACQUIRING); `s` screenshot; `q` quit. Every action
  that changes the sim is an event in the replay log.
- Window layer: side-by-side composited views (scaled to fit the screen; click coordinates mapped back to each
  view's full-resolution pixel), HUD from P0 plus selected object, its material and `requires_review` flag, ignition
  radius, last message. Real-time loop uses `sim.run_frame(wall_dt)` (fixed dt inside).
- `python -m livefire.app interactive --scene table_v1` runs on the synthetic scene (so the user can try clicking
  before cameras are calibrated).
- Tests (headless, controller-level): click on a paper pixel in each synthetic view -> world point within 1 cm of the
  analytic surface and paper elements start burning after one step; click on a room-wall/hole pixel -> no ignition,
  message `hole`; click on metal -> message `not flammable`; right-click + key `3` changes the object's material and
  the override is recorded; scripted click session -> replay digest matches; window coordinate mapping round-trips
  for both panes at a non-integer display scale.

**Acceptance:** tests pass; on a machine with a display, `interactive` opens and clicks start fire.

## Task 3: P1-003 ChArUco calibration with held-out validation

**Files:** `src/livefire/calib/{board.py,intrinsics.py,extrinsics.py,store.py,synthetic_board.py}`,
`src/livefire/app/calib_cli.py`, `configs/calibration.yaml`, `tests/test_calibration.py`, Makefile `calibrate`
target (replaces the P0 stub) + README rows.

- `configs/calibration.yaml`: board `squares_x 7, squares_y 5, square_length_m 0.035, marker_length_m 0.026,
  dictionary DICT_5X5_100` (fits A4 landscape at 100 % scale), `min_views 20`, `holdout_fraction 0.2`, seed.
  The user must measure the printed square size with a ruler and enter it (`square_length_m`), the marker length is
  scaled by the same factor.
- `board.py`: board object; `python -m livefire.app calib board --out board.png --dpi 300` writes a print-ready PNG
  (and states the physical size); detection helper returning ChArUco corners + ids per image.
- Intrinsics per camera: interactive capture `calib intrinsics --cam N` (window with detection overlay, coverage
  heat-map of corners over the image, auto-capture when the board is still and the pose differs from previous
  captures; `--from-dir` to use saved images). Fit with OpenCV (`calibrateCameraCharuco` or the 5.x equivalent),
  split views into fit/held-out with the seeded fraction, report RMS and median reprojection error on held-out views
  separately (PRD 7.1, target RMS < 1 px), save images + results.
- Extrinsics: board flat on the table visible from all cameras (`calib extrinsics`); per camera solvePnP on the
  ChArUco corners -> `T_board_from_camera`; convert to world per P1R2 (origin board centre, +Z up toward the
  cameras: flip if needed so cameras have z > 0); report per-camera reprojection RMS and the board-corner
  consistency between cameras (triangulate corners from both views, median distance to the known board layout).
- Store `data/calib/calib_vNNN.yaml` (+ images, manifest) with `calib_version` incremented, per camera K, dist,
  resolution, T_world_from_camera; `active` pointer file. Loaders produce `CameraCalibration` (P0) and undistortion
  maps (`cv2.initUndistortRectifyMap` with the same K as the undistorted K stored, alpha=0 policy documented).
- Synthetic board renderer (tests, MOCK-free ground truth): warp the board image into a camera with known K, dist,
  pose (homography for the plane + `cv2.projectPoints`-consistent distortion), add seeded noise/blur.
- Tests: intrinsics from 25 synthetic views recover fx, fy, cx, cy within 0.5 % and dist k1 within 0.02, held-out RMS
  < 0.5 px; extrinsics recover camera pose within 2 mm / 0.2 deg; world-frame convention: board centre maps to the
  origin, cameras above the table have z > 0, +Z toward cameras; version increments; round trip save/load; a
  calibration with a different resolution than the frames raises.

**Acceptance:** tests pass; the user-facing flow documented step by step in the README.

## Task 4: P1-004 Depth backends spike and ADR-0005

**Files:** `src/livefire/perception/depth/{base.py,anchor.py,hf_backends.py,oracle.py}`, `bench/depth_spike.py`,
`docs/decisions/ADR-0005-depth-backend.md`, `tests/test_depth_anchor.py`, `tests/test_depth_backends.py`,
`pyproject.toml` (`depth` extra + uv sources), `configs/depth.yaml`.

- Time-boxed research (PRD 0.2, max 3 candidates, web search for 2025-2026 options) for monocular depth that fits
  8 GB, runs at >= 2 fps at 720p on an RTX 4060 class GPU and has a usable license; seeds: Depth Anything V2
  (metric-indoor and relative), Depth Anything 3, MoGe-2, UniDepth V2, Depth Pro. Pick <= 3 that load through
  Hugging Face `transformers` or a pip package without custom CUDA builds.
- `DepthBackend` implementations per PRD 9 (warmup, infer -> `DepthResult` host NumPy per P0 ruling, latency and
  VRAM properties, fp16 on CUDA, model download cached under `data/models/`), selected by `configs/depth.yaml`.
  `OracleDepthBackend` (`# MOCK:` ground truth from the synthetic renderer, optional seeded noise/bias) for tests.
- `anchor.py` (P1R3): given a calibrated camera, the table-region mask (board footprint expanded, or a polygon in
  world coordinates) and a raw depth/disparity map, compute analytic plane depth for table pixels and fit
  scale (metric) or scale+shift in the model's native space (relative/disparity) with RANSAC + least squares;
  return anchored Z-depth, inlier ratio and residual statistics; reject (raise `DepthError`) below a configurable
  inlier ratio.
- `bench/depth_spike.py`: on (a) the synthetic scene rendered with the P0 renderer (sanity only), (b) a small public
  RGB-D sample with a permissive license (e.g. a few frames of TUM RGB-D or another dataset whose license allows
  research use; record license and source URL in a manifest; download into `data/`, gitignored) evaluated with
  table/floor-plane anchoring, and (c) the user's recorded clips when present: metric MAE, AbsRel, valid coverage,
  latency, peak VRAM; CSV in `bench/results/`. CPU timings labelled CPU.
- ADR-0005 scored by PRD 11.6, status **proposed** until the user's clips and GPU timings are in.
- Tests: anchoring recovers a known scale (and shift) from a synthetic plane with outliers within 1 %; rejection on
  low inliers; Oracle backend contract (shapes, invalid = 0.0, no NaN); HF backends tested only if the `depth` extra
  and weights are available (skip otherwise), on one small image, output contract only.

**Acceptance:** anchoring tests pass; spike CSV committed for what could be run; ADR drafted.

## Task 5: P1-005 Acquisition: accumulation, TSDF fusion, objects, snapshot

**Files:** `src/livefire/fusion/{accumulate.py,tsdf.py,extract.py,objects.py}`, `src/livefire/scene/acquire.py`,
`configs/acquisition.yaml`, `tests/test_fusion.py`, `tests/test_acquire.py`.

- Accumulation (PRD 7.2): per view over the acquisition window (`acquisition.seconds`, default 5 s, or N frames)
  per-pixel robust median of anchored depth with a confidence = fraction of frames agreeing within a relative
  tolerance; pixels below `min_confidence` invalid.
- TSDF fusion in Warp over an acquisition sub-ROI (`acquisition.fusion_roi`, default the table area
  x,y in [-0.8, 0.8] x [-0.6, 0.6], z in [-0.05, 0.6]) at `voxel_size 0.01 m`, truncation 3 voxels, weights from
  confidence; both views fused. Surface extraction: zero-crossing voxels -> surfel positions (interpolated) and normals
  (TSDF gradient), then the existing P0 surfel binning/area/neighbour code produces elements (reuse
  `scene/surfels.py`; no duplicated logic). Background beyond the fusion ROI is kept per view as measured depth for the
  known-clear hole policy (P0 ruling).
- Objects (P1R4): table plane RANSAC (agreement with z = 0 within `plane_tol_m 0.01`, else warn), clustering of the
  rest into objects, minimum object size, ObjectRegistry with per-camera observations; materials per P1R4.
- `acquire.py`: `acquire_snapshot(sources, calibration, depth_backend, cfg) -> SceneSnapshot` (undistort frames,
  infer depth per frame, anchor, accumulate, fuse, extract, cluster, build snapshot incl. `view_depth` by splatting
  and solids, timings per stage, acquisition report dict). Must work with 1 or 2 cameras.
- Tests: synthetic table_v1 through the full path with `OracleDepthBackend` + seeded noise (sigma 1 % of depth,
  per-view scale error 3 % before anchoring): table anchored to z = 0 within 5 mm; elements within 1 cm (p95) of
  analytic surfaces; 4 objects + table found with correct membership (> 90 % of each object's elements); snapshot
  passes all P0 snapshot validation; single-camera acquisition works; total acquisition time reported.

**Acceptance:** tests pass; acquisition on recorded synthetic clips (VideoFileSource + Oracle) under
`acquisition.seconds + 30 s` on CPU.

## Task 6: P1-006 Live mode end to end

**Files:** `src/livefire/app/live.py`, `src/livefire/app/change_monitor.py`, `src/livefire/render/geometry_overlay.py`,
wiring in `app/__main__.py`, `tests/test_live_pipeline.py`, Makefile `live` target + README rows (user runbook).

- `python -m livefire.app live [--cameras configs/cameras.yaml] [--calib active] [--depth <backend>]`:
  CALIBRATING (load active calibration; if the board is visible, verify reprojection < `verify_px`, else warn) ->
  ACQUIRING (Task 5, progress shown in the window) -> READY (release perception models, P1R6) -> SIMULATING with
  live undistorted frames composited every display frame and the Task 2 controller (click-to-ignite, tagging, keys).
- Geometry overlay (`o`): reconstructed elements projected onto each live view, coloured by object/material, as the
  alignment check (PRD 12 P1 task).
- Change monitor (PRD 5.1): compare live frames to the reference appearance captured at READY on static regions
  (outside current fire/smoke footprint), downscaled grayscale, robust per-pixel difference; if the changed fraction
  exceeds `change.threshold` for `change.frames` consecutive frames -> PAUSED with HUD "scene changed: press r to
  re-acquire"; never rewrites the world silently.
- Camera drop: a camera that stays `failed` for `drop_seconds` is removed from rendering (single-view fallback, HUD
  says so) and re-added when it reconnects.
- Latency per PRD 5.5 in HUD and logs: capture-to-display (`t_present - t_arrive`), frame age at use, sim lag,
  delivered fps; per-frame CSV in the run dir.
- Tests (headless): synthetic clips as `VideoFileSource`s + Oracle depth + stored synthetic calibration -> live
  pipeline reaches SIMULATING, a scripted click ignites the paper, composited frames written; change monitor triggers
  on an injected scene change clip and not on a static clip with sensor noise; camera drop and recovery handled;
  latency columns present.

**Acceptance:** tests pass; README runbook covers the full physical workflow.

## Task 7: P1-007 Gate 1 evaluation tools and camera placement study

**Files:** `eval/{alignment.py,geometry_mae.py,placement.py,latency.py}`, `src/livefire/app/measure_cli.py`,
`docs/RUNBOOK_P1.md`, `tests/test_eval_p1.py`.

- M6 alignment (`eval/alignment.py`): with the board (or AprilTags at known positions) visible after acquisition,
  compare reconstructed landmark positions (from the fused geometry along each landmark's pixel ray) with known
  world positions; median/p95 error CSV.
- M4 geometry MAE (`measure` CLI): the user clicks pairs of points in a view (snapshot back-projection) and types the
  tape-measured distance; the tool stores the pairs and reports MAE in cm; also AbsRel/completeness on synthetic.
- Placement study (`placement.py`, PRD 7.1): from calibration + first acquisition: angle between cameras, shared
  visible table area fraction, depth-valid coverage per view; fail loudly with concrete guidance if below thresholds
  (`placement.min_shared_fraction`, `min_valid_coverage`, angle outside 40-100 deg).
- Latency eval: parse the live per-frame CSV into M2 statistics; glass-to-glass procedure doc.
- `docs/RUNBOOK_P1.md`: print board, measure squares, place cameras (60-90 deg apart, 1-1.5 m from table), phone IP
  camera setup, calibrate intrinsics per camera, extrinsics, run live, tag materials, click to ignite, run evals,
  which files to send back.
- Tests on synthetic data for each eval (known answers).

**Acceptance:** tests pass; runbook complete.

## Task 8: P1-008 Phase 1 report (controller)

`docs/reports/phase-1.md` and STATUS from committed artifacts; Gate 1 needs the user's physical runs (M4, M6
measured on a real scene; fire on real geometry with manual materials; baseline performance on the 4060).
