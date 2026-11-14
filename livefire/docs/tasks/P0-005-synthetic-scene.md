## Task 5: P0-005 Procedural synthetic two-camera scene

**Files:** `src/livefire/scene/synthetic.py`, `src/livefire/capture/synthetic_source.py`, `tests/test_synthetic.py`.

- Analytic primitives (NumPy ray casting, vectorised over all pixels): axis-aligned boxes (slab method) and
  rectangles (finite planes, two-sided). Each primitive has `obj_id`, `name`, `material` (name from MaterialTable),
  linear albedo RGB.
- Scene "table_v1" (world metres):
  - obj 0 `table`: box x [-0.8, 0.8], y [-0.5, 0.5], z [-0.05, 0.0], `other_nonflammable`, albedo (0.45, 0.33, 0.22).
  - obj 1 `wood_block`: box x [0.125, 0.275], y [0.07, 0.17], z [0.0, 0.08], `wood`, albedo (0.55, 0.38, 0.2).
  - obj 2 `metal_box`: box x [-0.34, -0.26], y [0.16, 0.24], z [0.0, 0.08], `metal`, albedo (0.6, 0.6, 0.62).
  - obj 3 `paper_sheet`: vertical rectangle in plane y = 0.0, x [-0.15, 0.0], z [0.0, 0.2], two-sided, `paper`,
    albedo (0.85, 0.85, 0.82).
  - obj 4 `cloth_patch`: horizontal rectangle z = 0.003, x [-0.2, 0.05], y [-0.25, -0.02], upward-facing, `cloth`,
    albedo (0.6, 0.15, 0.15).
  - Backdrop (outside the ROI, so it is never reconstructed as fuel): obj -1 `room`: floor plane z = -0.75 and walls
    at x = +-2.5, y = +-2.5 (finite rectangles spanning the room), albedo (0.5, 0.5, 0.5), material none.
    Rays that hit nothing have depth 0.0, valid False, black RGB.
- Cameras from `configs/camera.yaml` via `look_at` (calib_version from the file).
- `render_view(scene, calib, seed=None) -> SyntheticView(rgb uint8 sRGB, rgb_linear f32, depth f32 Z-depth,
  valid bool, obj_id i32 (-1 background/room/miss), normal_world f32)`. Shading: Lambert with one directional light
  direction normalize(0.3, -0.4, 1.0) intensity 0.8 plus ambient 0.2, in linear light, then sRGB encode (IEC 61966-2-1).
  Optional seeded Gaussian sensor noise `noise_sigma` (default 0). Deterministic.
- `SyntheticCameraSource(CameraSource)`: re-emits the static rendered frame from a background thread at a configured
  fps with `t_arrive_ns = perf_counter_ns()`, increasing `seq`, keeps only the newest frame.
- Ground-truth helpers: `scene.objects_table()` (obj_id -> name, material), analytic visible area per object
  per view is NOT required; analytic surface distance function `scene.distance_to_object(obj_id, p (N,3))` and seeded uniform surface
  sampling `scene.sample_surface(obj_id, n, rng) -> (points (n,3), normals (n,3), total_area_m2)` are required
  (for Task 6 tests).
- Tests: for 500 seeded pixels with valid depth, back-projected points lie on their primitive
  (`distance_to_object < 1e-4 m`); depth equals the analytic intersection; all objects 0..4 are visible in both views
  (>= 200 pixels each); determinism (two renders bitwise equal); `SyntheticCameraSource.latest()` returns
  increasing seq and drops stale frames.

**Acceptance:** tests pass; `python -m livefire.scene.synthetic --out out/synthetic` writes the two RGB, depth (16-bit
PNG mm) and obj_id visualisations for inspection.

