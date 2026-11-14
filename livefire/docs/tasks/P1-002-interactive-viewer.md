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

