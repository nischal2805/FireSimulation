## Task 3: P0-003 Interface contracts and material table

**Files:** `src/livefire/common/interfaces.py` (Protocols), `src/livefire/capture/types.py` (`Frame`),
`src/livefire/perception/types.py` (`DepthResult`, `SegmentationResult`), `src/livefire/perception/material/types.py`
(`MaterialPrediction`), `src/livefire/scene/snapshot.py` (`SceneSnapshot`, `ObjectRegistry`, `WorldObject`),
`src/livefire/sim/state.py` (`FireStateHost`), `src/livefire/sim/materials.py` (`MaterialTable`),
`tests/test_contracts.py`, `tests/test_materials.py`.

- Protocols (PRD 9), each with a docstring that states shapes, dtypes, units, coordinate frame, device placement,
  contiguity, sync requirement, ownership, error behaviour and `calib_version` tie: `CameraSource`, `CameraCalibration`
  (re-export the Task 2 dataclass; no new Protocol class needed beyond a type alias), `DepthBackend`,
  `SegmentationBackend`, `MaterialBackend`, `SceneReconstructor`, `SimulationBackend`, `RendererBackend`,
  `ReplayController`, `BenchmarkRecorder`. Every backend Protocol has `latency_ms`, `peak_vram_allocated_mb`,
  `peak_vram_reserved_mb` properties (PRD 9). Use `DepthBackend` text from PRD 9 as the model.
  `SimulationBackend`: `reset(snapshot) -> None`, `ignite(point_world (3,), radius_m) -> int` (n elements lit),
  `step(n_steps: int) -> None` (fixed `dt_sim`), `state_host() -> FireStateHost`, `t_sim: float`, `step_count: int`,
  `set_material(obj_id, material_id) -> None`.
  `RendererBackend`: `render(cam_id: int, frame: Frame, fire: FireStateView) -> ndarray uint8 (H,W,3) sRGB`
  composited, where `FireStateView` is a Protocol exposing the sim backend's device-resident read-only fields
  (`T_g`, `soot`, `char`, `removed`, `h_gas`, `roi`) — the renderer never writes simulation state.
  `CameraSource`: `start()`, `stop()`, `latest() -> Frame | None` (newest-frame-wins), `cam_id`.
- `Frame`: `cam_id, rgb uint8 (H,W,3) undistorted sRGB, t_arrive_ns: int, seq: int, calib_version: int`.
- `DepthResult`: `z_depth_m f32 (H,W)`, `valid bool`, `conf f32 [0,1]`, `calib_version`, invalid = 0.0; `__post_init__`
  rejects NaN and nonzero invalid pixels (raise `ValueError`).
- `MaterialPrediction` exactly as PRD 7.3 (`class_probabilities f32[n_materials]`, `category_hint`, `confidence`,
  `source: Literal["model","category_prior","manual_override"]`, `requires_review`); validate probabilities sum to 1
  (tol 1e-4).
- `SceneSnapshot` exactly the PRD 6.1 field set (frozen dataclass) with these dtypes: pos/normal f32 (N_e,3), area f32,
  obj_id i32 (-1 = unknown), material_id u8, material_conf f32, neighbors i32 (N_e,K) -1 padded, solid_mask u8 (G,G,G),
  solid_fraction f32 (G,G,G), view_depth f32 (H,W) per cam, view_hole_mask bool per cam, objects: ObjectRegistry.
  Add `surfel_spacing: float` (needed by conduction; record as an additive field in the report). `__post_init__`:
  validates shapes/dtypes, sets every array `flags.writeable = False`, and checks every camera's `calib_version`
  equals `snapshot.calib_version` (raise `CalibVersionError`). Helper `assert_calib_version(snapshot, frame)`.
- `ObjectRegistry` / `WorldObject(obj_id, name, material: MaterialPrediction, observations: dict[cam_id, int])`:
  minimal; `override_material(obj_id, material_name)` stores `source="manual_override"`.
- `FireStateHost` (NumPy copy of PRD 6.2 fields: T_s, ignition_exposure, m_fuel, char, removed, burning u8,
  T_g, vel (G,G,G,3) cell-centred, soot, p, t_sim, step) with `digest(decimals=6) -> str` (sha256 of rounded arrays,
  used for replay).
- `MaterialTable.load(path)` reads `configs/materials.yaml`; class order fixed: paper, cloth, wood, plastic, metal,
  glass, ceramic, other_nonflammable -> `material_id` 0..7. Validates: flammable entries have all fuel fields
  (`m0_areal, T_ign, t_delay, mdot_ref, dH, chi, char_yield, y_soot, consumed_when_burnt`); non-flammable entries
  have no fuel fields (error if present); every entry has `rho, c, k, delta, eps, status`; positive values;
  `0 <= chi, char_yield, eps <= 1`. Exposes per-id NumPy arrays (`rho, c, k, delta, eps, flammable, m0, m_res,
  T_ign, t_delay, mdot_ref, dH, chi, y_soot, consumed`) with non-flammables having `m0 = m_res = 0`.
  `C_areal = rho * c * delta`.
- Tests: dataclass validation (bad dtype/shape, NaN depth, nonzero invalid depth, prob sum), snapshot arrays read-only,
  calib mismatch raises, material table ids and derived arrays (paper `m_res == 0`, wood `m_res == 0.15 * 3.0`),
  non-flammable with fuel field rejected, `FireStateHost.digest` stable and sensitive to a 1e-3 change.

**Acceptance:** tests pass; every Protocol docstring covers the PRD 9 checklist.

