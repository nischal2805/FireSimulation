# P1-006 Live mode end to end (task brief, copied from the controller plan)

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

