# P0 implementation plan (contracts and synthetic vertical slice)

Spec: `docs/PRD.md` (authority). Rules: `CLAUDE.md`. Approved by the user on 2026-11-14 together with
resolutions 1-11 listed in "Approved rulings" below.

## Global Constraints (bind every task)

- Project root: `/home/user/FireSimulation/livefire` (git repo root is its parent `FireSimulation/`).
  Package `src/livefire/`, tests `tests/`, configs `configs/`, bench `bench/`, eval `eval/`, `data/` gitignored.
- Python **3.11 exactly** (`requires-python = "==3.11.*"`), env managed with `uv` (`uv.lock` committed). Run tools via
  `uv run ...` from the project root. `make test` = `uv run pytest -m "not slow" -q`; `make test-slow` = all tests.
- No hand-written CUDA C++. GPU kernels in NVIDIA Warp (ADR-0001: Warp primary, Taichi fallback; status proposed pending the GPU run).
  This container has **no GPU**: kernels must run on the framework's CPU device; device is selected by
  `configs/app.yaml: device` (`"cpu"` here, `"cuda:0"` on the user's machine). Never hard-code `cuda`.
- Geometry contract (PRD section 4), verbatim: world right-handed, metres, +Z up, origin at board centre on the
  table plane; camera frame OpenCV (+X right, +Y down, +Z forward); pose `T_world_from_camera` (4x4, camera->world);
  name every transform `T_<to>_from_<from>`; `K` 3x3 pixels for the stored resolution; Z-depth along camera +Z in metres
  (not ray distance); invalid depth pixels hold `0.0` with `valid=False`, never NaN; voxel centre
  `roi_min + (i + 0.5) * h`, `h = roi_size / N`; every snapshot records its `calib_version`, mismatch is an error.
  Integer pixel coordinates `(u, v)` denote pixel centres (OpenCV convention). Back-projection:
  `p_c = d * inv(K) @ [u, v, 1]^T`, `p_w = T_world_from_camera @ [p_c, 1]^T`.
- All tunables in YAML under `configs/`. Physical/artistic parameters use the form
  `name: {value: <v>, unit: "<u>", range: [lo, hi], status: physical_estimate|artistic}` (PRD 8.1).
  `configs/materials.yaml` keeps the PRD 8.5 format exactly.
- Seed all randomness (`numpy.random.default_rng(seed)`; seed from `configs/app.yaml: seed`).
- Never fabricate numbers. Any mock or placeholder is marked `# MOCK:` in code and labelled in output.
  Benchmarks run here are CPU numbers and must be labelled `gpu=none, framework=<fw>-cpu`.
- Use only accessor helpers from `livefire.common.geometry` for voxel index/centre maths.
- Each task: first copy your brief verbatim to `docs/tasks/<P0-ID>-<slug>.md`; at the end update
  `docs/STATUS.md` (completed tasks, next task, blockers) and commit with message `P0-00N: <title>` plus trailers:
  a `Co-Authored-By: Claude <your model name> <noreply@anthropic.com>` trailer and
  `Claude-Session: https://claude.ai/code/session_01FvsiJ1w9gRFcZuiYu1oxY3`. Do not push.
- `ruff check` clean, type hints on public functions, docstrings on public interfaces stating shapes, units, frame,
  device, ownership.

## Approved rulings (from the plan approval)

R1 `livefire/` is a subfolder of the git repo. R2 no GPU here: ADR-0001 stays "proposed" until the user runs
`make bench-spike` on the RTX 4060; CPU CSVs are labelled CPU. R3 P0 surface elements come from back-projected
synthetic depth binned into surfels (TSDF in P1, same output format). R4 gas velocity is stored staggered (MAC)
internally; `FireState.vel` cell-centred `[G,G,G,3]` is exposed through an accessor (ADR-0002). R5 combustion energy:
`q_release = chi * dH * mdot''`; fraction `f_feedback` (sim.yaml, artistic) heats the surface, `1 - f_feedback`
goes to the gas; add surface re-radiation loss `eps * sigma * (T^4 - T_amb^4)`; clamp `T_s <= T_s_max`,
`T_g <= T_g_max` and count clamp events (ADR-0003). R6 conduction neighbours only within the same `obj_id`; the
synthetic table is `other_nonflammable`. R7 a lit element burns until burnt out; `T_pilot`, ignition radius in
config. R8 default ROI `roi_min=(-1.5,-1.5,-0.1)`, `roi_max=(1.5,1.5,2.9)`. R9 the resolution-sensitivity
invariant and the all-voxel ablation are P2; the test exists in P0 but does not gate. R10 renderer kernels use the
ADR-0001 framework; headless run by default. R11 one commit per task.

Controller rulings made while writing this plan (ledgered):
- Conduction is computed as symmetric **edge fluxes** so energy is conserved exactly:
  `P_ij = G_ij (T_j - T_i)` [W], `G_ij = kd_ij * s / d_ij`, `kd_ij` = harmonic mean of `k_m * delta_m` of i and j,
  `d_ij` = distance, `s` = surfel spacing. `dT_i = dt * sum_j P_ij / (A_i * C''_i)`. On a uniform grid with
  `A = s^2, d = s` this is exactly PRD 8.2's `(k delta / s^2) * sum (T_j - T_i)`. Neighbour lists must be symmetric.
  Explicit stability: `dt_cond_max = 0.5 / max_i(sum_j G_ij / (A_i C''_i))`; the model sub-steps when `dt` exceeds it.
- `SceneSnapshot` arrays are host NumPy arrays marked read-only; sim and render backends create their own device
  mirrors once at READY and own them. (PRD 6.1 comments say "on GPU"; this is the ownership statement.)
- Gas temperature and heat are sampled/injected at `x_i + n_i * h_gas` (one gas cell along the element normal),
  so elements on solid cells still see fluid.
- Hole policy for occlusion: `view_depth` = splat of surface elements; where no element covers a pixel but the
  accumulated measured depth is valid and lies beyond the ROI exit along that ray, `view_depth` = that measured
  depth (known-clear background); otherwise the pixel is a hole (`view_hole_mask=True`) and flames are suppressed
  there.
- Long-run invariant (10 min simulated) runs at G=32 in this container (slow marker); G=64 is part of
  `make test-slow` on the GPU machine.

---

## Task 1: P0-001 Repo skeleton, environment, config loader, timers

**Files:** `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `.gitignore`, `configs/{app,grid,camera,sim,render,materials}.yaml`,
`src/livefire/__init__.py`, `src/livefire/{common,capture,perception/depth,perception/segment,perception/material,fusion,scene,sim,render,app}/__init__.py`,
`src/livefire/common/{config.py,timers.py,log.py}`, `tests/test_config.py`, `tests/test_timers.py`, `bench/{results,plots,traces}/.gitkeep`, `eval/.gitkeep`.

- `pyproject.toml`: name `livefire`, src layout, `requires-python = "==3.11.*"`. Runtime deps: numpy, scipy, pyyaml,
  opencv-python-headless, warp-lang, matplotlib. Optional group `spike`: taichi. Dev group: pytest, ruff.
  pytest config registers marker `slow`. Ruff line length 110.
- `uv lock` and `uv sync --all-groups` must work; commit `uv.lock`.
- `.gitignore`: `data/`, `.venv/`, `__pycache__/`, `out/`, `*.mp4` under `out/`, `.superpowers/`, `bench/traces/*` (keep .gitkeep).
- Makefile targets: `setup` (`uv sync --all-groups`), `test`, `test-slow`, `lint` (`uv run ruff check src tests bench`),
  `run`, `bench`, `bench-spike`, `plots`, `calibrate`, `eval`. Targets for later tasks call fixed paths:
  `run` -> `uv run python -m livefire.app run --scene table_v1 --seconds 8 --ignite "-0.075,0.0,0.05@0.5" --out out/demo`,
  `bench` -> `uv run python bench/run_bench.py`, `bench-spike` -> `uv run python bench/spike/run_spike.py`,
  `plots` -> `uv run python bench/plots.py` (these files do not exist yet; that is expected); `calibrate` and `eval` print
  `"not implemented until P1"` and exit 2.
- `configs/app.yaml`: `seed: 1234`, `device: "cpu"`, `dt_sim: 0.0166666667` (1/60 s), `max_substeps_per_frame: 4`,
  `headless: true`, `log_level: INFO`.
- `configs/grid.yaml`: `G: 96`, `roi_min: [-1.5, -1.5, -0.1]`, `roi_max: [1.5, 1.5, 2.9]`, `surfel_spacing: 0.015`,
  `neighbor_radius_factor: 1.5`, `neighbor_k_max: 16`.
- `configs/camera.yaml`: synthetic rig: width 1280, height 720, fx=fy=900.0, cx=639.5, cy=359.5, dist zeros(5),
  `calib_version: 1`; cam0 position [1.0, -0.9, 0.7], cam1 position [-0.6, -1.1, 0.65], both target [0.0, 0.0, 0.08].
- `configs/materials.yaml`: PRD 8.5 table verbatim (with the units comment).
- `configs/sim.yaml` and `configs/render.yaml`: create with a header comment describing the
  `{value, unit, range, status}` form and one example-free empty mapping (`{}`); Tasks 7, 8, 9 add their parameters.
  The loader must accept an empty file/mapping.
- `common/config.py`: `load_yaml(name) -> dict` (from `configs/<name>.yaml`, root found relative to the package or via
  env `LIVEFIRE_CONFIG_DIR`); `Params` wrapper: `params = load_params("sim")`, `params["h_c"] -> float` returns `value`,
  validates every entry has `value, unit, range, status` with status in `{physical_estimate, artistic}` and
  `range[0] <= value <= range[1]`; raises `ConfigError` with the key name otherwise. `Params.meta(key)` returns the dict.
- `common/timers.py`: `StageTimer` using `time.perf_counter_ns`; `with timer.stage("name"):`; `timer.record(name, ms)`;
  `timer.summary() -> dict[name, {"n", "median_ms", "p95_ms", "p99_ms"}]`; `timer.reset()`. A `device_sync`
  callable can be passed to `stage(...)` to synchronise before reading the clock (GPU boundary rule PRD 5.5).
- `common/log.py`: `get_logger(name)` standard logging with level from app.yaml.
- Tests: config loads all six files; a bad status, missing unit, out-of-range value each raise `ConfigError`;
  timer percentiles are right on a known list (`record` 1..100 ms -> median 50.5, p95 ~95.05 using numpy percentile
  linear); `stage` measures a `time.sleep(0.01)` as >= 9 ms.

**Acceptance:** `make setup && make test && make lint` green.

## Task 2: P0-002 Geometry module and tests

**Files:** `src/livefire/common/geometry.py`, `tests/test_geometry.py`.

API (NumPy, float64 internally, vectorised over N):
- `make_K(fx, fy, cx, cy) -> ndarray(3,3)`.
- `@dataclass(frozen=True) class CameraCalibration`: `cam_id: int, width: int, height: int, K: ndarray(3,3),
  dist: ndarray(5,), T_world_from_camera: ndarray(4,4), calib_version: int`; property `T_camera_from_world`;
  property `center_world -> (3,)`; `__post_init__` validates shapes and that the rotation is orthonormal with det +1
  (tol 1e-6). `K` here is the undistorted K.
- `invert_transform(T) -> T_inv` (rigid inverse, not `np.linalg.inv`); `transform_points(T_b_from_a, p_a) -> p_b`.
- `look_at(position, target, up=(0,0,1)) -> T_world_from_camera`: z = normalize(target - position),
  x = normalize(cross(z, up)), y = cross(z, x); rotation columns `[x y z]`. (Gives OpenCV +Y down.)
- `project(calib, p_world (N,3)) -> (uv (N,2), z (N,), valid (N,) bool)`; z is camera Z; points with z <= 0 get
  `uv = 0.0` and `valid = False` (no NaN).
- `backproject(calib, uv (N,2), z (N,)) -> p_world (N,3)` per the contract formula.
- `pixel_grid(width, height) -> (u, v)` integer centre grids; `backproject_depth(calib, depth (H,W), valid (H,W))
  -> points (H,W,3)` with invalid pixels = 0.0.
- `ray_dirs_camera(calib) -> (H,W,3)` = `inv(K) @ [u,v,1]` (z component 1); `zdepth_to_raydist(calib, z)` and
  `raydist_to_zdepth(calib, r)` per pixel (H,W), using `|inv(K)[u,v,1]|`; invalid (0.0) stays 0.0.
- `@dataclass(frozen=True) class AABB(min, max)` with `size`, `center`, `contains(p) -> bool[N]`,
  `ray_intersect(origins (N,3), dirs (N,3)) -> (t_near, t_far, hit)` (slab method; no NaN, misses have hit=False).
- `@dataclass(frozen=True) class VoxelGrid(roi: AABB, N: int)`: requires a cubic ROI (raise ValueError otherwise);
  `h = roi.size[0] / N`; `centers(idx (M,3) int) -> (M,3)`; `index_of(p (M,3)) -> idx int (M,3)` (floor);
  `in_bounds(idx) -> bool[M]`; `linear_index(idx)`/`unravel(lin)` in C order `(ix, iy, iz)`.
- Tests (PRD 4, no camera/AI): (a) two synthetic cameras with asymmetric poses (different positions, a 17 deg roll on
  one, non-centred cx/cy e.g. 655.3/341.9) and 200 seeded random off-axis points inside a 1 m cube in front of both:
  project then back-project, max error < 1e-4 m; (b) a transform-direction flip test: using `T_camera_from_world` in
  place of `T_world_from_camera` makes the round trip error > 0.1 m (assert the wrong path fails); (c) a hand-computed
  case: camera at (0,0,1) looking at the origin with `look_at`, the origin projects to (cx, cy) and world +X maps to
  image +u; (d) Z-depth vs ray distance: at a corner pixel the ratio equals `|inv(K)[u,v,1]|` and round trips; centre
  pixel ratio ~1; (e) voxel index/centre round trip at all 8 grid corners and random interior points, and points
  exactly on `roi_min` map to index 0; (f) `AABB.ray_intersect` on hits, misses and an origin inside the box;
  (g) `CameraCalibration` rejects a non-orthonormal rotation.

**Acceptance:** all tests pass; `ruff` clean.

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

## Task 6: P0-006 Snapshot builder (surface elements, obstacles, per-view occlusion)

**Files:** `src/livefire/scene/builder.py`, `src/livefire/scene/surfels.py`, `src/livefire/scene/splat.py`,
`tests/test_builder.py`.

Input: per view `(calib, depth, valid, obj_id, rgb)` plus `obj_id -> material` (for P0 from synthetic ground truth,
`MaterialPrediction.source = "manual_override"`, conf 1.0). Output: a `SceneSnapshot` (Task 3).

- **Normals from depth** (no analytic normals): back-project to world points; normal = normalized cross product of
  central differences (right-left, down-up); discard pixels where either difference spans a depth jump
  > `0.03 * depth` or crosses an `obj_id` change; orient the normal towards the camera centre.
- **Surfel binning** (spacing `s = grid.surfel_spacing`): key = `(obj_id, floor((p - roi_min) / s))` for points inside
  the ROI with `obj_id >= 0`, both views pooled. Per bin: `pos` = mean point, `normal` = normalized mean normal,
  `area` = for each view the sum of pixel footprint areas `(z/fx)*(z/fy)/max(|cos theta|, 0.2)` of that view's pixels
  in the bin, then max over views, capped at `2 s^2`. Drop bins with fewer than 3 points.
- **Neighbours:** radius search (`scipy.spatial.cKDTree`) with `r_nb = neighbor_radius_factor * s`, same `obj_id`
  only, excluding self, capped at `neighbor_k_max` nearest; then symmetrise: keep edge (i,j) only if j in N(i) and
  i in N(j); `-1` padding.
- **Obstacles at G = grid.G:** `solid_mask` = 1 for gas cells containing any element of an object whose material
  `delta >= 1e-3 m` (thick objects: table, wood, metal); thin objects (paper, cloth) are not obstacles.
  `solid_fraction = min(1, sum element area in cell / h^2)` for those cells, else 0.
- **Per-view occlusion depth (`splat.py`):** splat each element as a camera-facing disc of radius
  `1.25 * sqrt(area / pi)` into each view with a z-buffer (min Z-depth, disc depth = element centre Z); the hole
  policy from the controller rulings fills known-clear background from the measured depth; `view_hole_mask` = no
  element and not known-clear. Vectorise (group elements by footprint size); no per-pixel Python loops.
- `build_snapshot(views, materials, grid_cfg, version=1) -> SceneSnapshot` and a CLI
  `python -m livefire.scene.builder --scene table_v1 --out out/snapshot.npz`.
- Tests on scene table_v1: every element within `0.5 * s` of its object's analytic surface (95th percentile) and
  within `s` (max); every object 0..4 has elements; per-object total area within 20 % of the analytic area of the part
  visible from at least one camera (estimate the analytic visible area by Monte-Carlo sampling the object surface and
  testing visibility against the rendered depth of each view, seeded); neighbour lists symmetric, same obj, no self;
  materials match ground truth; `view_depth` vs ground-truth depth on object pixels with valid depth and no edge:
  median abs error < 1 cm, p95 < 3 cm; hole fraction over pixels that see objects 0..4 < 2 %; room-wall pixels are
  known-clear (not holes); the wood block's interior cell is not required to be solid but its surface cells are;
  paper cells are not solid; `snapshot.calib_version == 1`; arrays read-only.

**Acceptance:** tests pass; build time for table_v1 printed (CPU, reported as-is).

## Task 7: P0-007 Surface fire model (thermal, ignition, combustion, char)

**Files:** `src/livefire/sim/surface.py` (kernels + `SurfaceFireModel`), `src/livefire/sim/device.py` (device
selection and snapshot upload helpers), `configs/sim.yaml` (surface section), `tests/test_surface_invariants.py`.
Framework: **NVIDIA Warp** (`warp-lang`, ADR-0001, status proposed pending the GPU run). Warp's CPU backend is
single-threaded: keep test grids/element counts small (G <= 32 in unit tests) so `make test` stays fast.

- `SurfaceFireModel(snapshot, materials: MaterialTable, params, device)`: owns device arrays T_s, E, m_fuel, char,
  removed, burning, per-element material id (overridable), precomputed edge conductances `G (N,K)` (controller
  ruling formula), `Cap_i = A_i * C''_i`.
- `step(dt, T_g_sample (N,) device array) -> (heat_W (N,), soot_kg_s (N,))`: per element, in this order:
  1. thermal: `q_conv = h_c (T_g - T_s)`; conduction edge fluxes; `q_rr = -eps sigma (T_s^4 - T_amb^4)`;
     `q_fb = f_feedback * chi dH mdot''` while burning; `T_s += dt * (A (q_conv + q_rr + q_fb) + P_cond) / Cap`;
     clamp to `[T_amb_min, T_s_max]`, count clamps.
     Internal sub-stepping when `dt > dt_cond_max` (ruling) or when `dt * A h_c / Cap > 0.5` (convective stability);
     the number of sub-steps is exposed.
  2. ignition exposure per PRD 8.3 (`E += dt` if `T_s >= T_ign` else `E = max(0, E - decay dt)`); ignite when
     `E >= t_delay` and `m_fuel > m_res + eps_m` and flammable.
  3. combustion per PRD 8.3: `m_new = max(m_res, m_fuel - mdot_ref dt)`, `consumed = m_fuel - m_new`;
     `heat_W = (1 - f_feedback) * chi * dH * consumed * A / dt`; `soot_kg_s = y_soot * consumed * A / dt`;
     char per PRD formula (clamped 0..1; for `char_yield = 0` use `(m0 - m)/m0`); burnt out when
     `m_fuel <= m_res + eps_m` -> burning = 0; `removed = char if consumed_when_burnt else 0`.
  4. Non-flammables: never ignite, `m_fuel = 0` always, char/removed stay 0.
- `ignite(point, radius)`: elements within radius (flammable only) get `T_s = T_pilot`, `E = t_delay`; returns count.
- `sample_points()` returns `x_i + n_i * h_gas` (gas sampling/injection points, ruling).
- `configs/sim.yaml` surface params: `T_amb 293.15 K [250, 320] physical_estimate`; `h_c 25 W/(m^2 K) [1, 200]
  physical_estimate`; `f_feedback 0.3 - [0, 1] artistic`; `T_pilot 900 K [400, 1500] artistic`;
  `ignite_radius 0.03 m [0.005, 0.2] artistic`; `ignition_decay 0.5 1/s [0, 10] artistic`;
  `T_s_max 1500 K [800, 2500] artistic`; `T_s_min 250 K [200, 300] artistic`; `eps_m 1e-6 kg/m^2 [0, 1e-3] artistic`.
- Invariant tests (PRD 8.6), with T_g given as a constant or a prescribed field (no gas solver yet), on the Task 6
  snapshot and on tiny hand-built element graphs:
  fuel bounds `m_res <= m_fuel <= m0` and char/removed in [0,1] over a 120 s burn after ignition of paper;
  no spontaneous fire (T_g = T_amb, 600 s, nothing burns, T_s stays T_amb within 1e-3 K);
  non-flammables (metal, glass, ceramic, other) never ignite even at `T_g = 1500 K` for 60 s and after `ignite()`;
  conduction energy conservation on an isolated graph (h_c = 0, eps = 0 via test params, no burning): total
  `sum Cap_i T_i` conserved to 1e-6 relative over 1000 steps, and the temperature variance decreases monotonically;
  stability: large dt triggers sub-stepping and stays bounded;
  monotonicity smoke test: same geometry, paper vs cloth vs wood under the same T_g ramp ignites in order
  paper < cloth < wood (reports times; this is the M9 seed, not the P2 validation);
  determinism: two runs bitwise equal on CPU.

**Acceptance:** all invariant tests pass.

## Task 8: P0-008 Gas solver baseline and coupled simulation

**Files:** `src/livefire/sim/gas.py` (`GasSolver`, MAC grid), `src/livefire/sim/coupled.py` (`FireSimulation`
implementing `SimulationBackend`), `configs/sim.yaml` (gas section), `tests/test_gas.py`, `tests/test_coupled.py`,
`docs/decisions/ADR-0002-mac-grid.md`, `docs/decisions/ADR-0003-energy-split.md`.

- MAC grid G^3, `h = roi_size / G`: faces u (G+1,G,G), v (G,G+1,G), w (G,G,G+1); cell-centred T_g, soot, p,
  solid (from snapshot). Boundaries: domain walls and solid cells are no-through (face velocity 0 on faces touching
  a solid cell or a wall), except the top face (z = roi_max) which is open: pressure Dirichlet p = 0 and outflow
  allowed.
- Per step (`dt`): (1) advect u, v, w, T_g, soot semi-Lagrangian, RK2 backtrace, trilinear, clamp-to-domain;
  (2) add sources: `T_g += heat_J / (rho_g c_p h^3)` and `soot += soot_kg / h^3` scattered trilinearly at the
  element sample points (deterministic on CPU; on GPU atomics allowed, replay tolerance stated);
  (3) buoyancy on w faces `a_z = g (T_face - T_amb) / T_amb`, `g = 9.81`; (4) vorticity confinement strength
  `eps_vc`; (5) wind: inflow velocity on the -x wall plus seeded low-amplitude noise, default off (zero);
  (6) velocity clamp `|u| <= cfl_max h / dt` with a violation counter; (7) pressure projection, red-black
  Gauss-Seidel with `n_iters` iterations, solids as Neumann boundaries; record `||div u||_2` over fluid cells before
  and after; (8) cooling `T_g = T_amb + (T_g - T_amb) exp(-k_cool dt)`, soot `*= exp(-k_soot dt)`, clamp
  `T_g <= T_g_max` with counter. Accessor `velocity_cell_centered() -> (G,G,G,3)`.
- `configs/sim.yaml` gas params: `rho_g 1.2 kg/m^3 physical_estimate`, `c_p 1005 J/(kg K) physical_estimate`,
  `g 9.81 m/s^2`, `eps_vc 0.3 - artistic [0, 2]`, `k_cool 1.5 1/s artistic [0, 20]`, `k_soot 0.3 1/s artistic [0, 10]`,
  `T_g_max 2000 K artistic [1000, 3000]`, `cfl_max 3.0 - artistic [0.5, 10]`, `n_iters 60 - artistic [5, 500]`,
  `wind_speed 0.0 m/s artistic [0, 5]`, `wind_noise 0.0 m/s artistic [0, 1]`. Ranges for those not given: choose
  sensible ones and list them in the report.
- `FireSimulation(snapshot, materials, sim_params, app_cfg)`: fixed `dt_sim`; per `step`: sample T_g at element sample
  points (trilinear), `surface.step`, inject sources, `gas.step`. `run_frame(wall_dt)` advances
  `min(max_substeps_per_frame, accumulated / dt_sim)` steps (accumulator; never variable dt) and returns
  `rt_factor` (simulated s / wall s over a sliding window) and `sim_lag_s`. `ignite`, `set_material`,
  `state_host()` (FireStateHost incl. cell-centred vel), `stats()` (clamp counters, divergence norms, substeps).
- Tests: projection reduces `||div||_2` by >= 10x on a seeded random field at G=32 with default `n_iters`
  (report the actual factor; if 10x needs more iterations, report and set the default accordingly); no flux through
  solid faces after projection (exactly 0); zero-wind symmetry: a hot blob at the domain's x-y centre, no solids,
  60 steps at G=32: centre of mass of `T_g - T_amb` drifts < 0.25 h in x and y, and the field is mirror-symmetric
  in x within 1e-3 relative L2; buoyancy: a hot blob rises (centre of mass z increases); long run (slow marker):
  coupled sim on a down-scaled table_v1 snapshot at G=32, paper ignited, 10 min simulated (36000 steps): no NaN/Inf,
  all fields bounded (T_g <= T_g_max, |u| bounded by the clamp, soot >= 0), invariants of Task 7 still hold;
  replay: two coupled runs with the same seed and events give equal `FireStateHost.digest()` on CPU;
  resolution sensitivity (R9, non-gating, `xfail(strict=False)` if it fails): ignition time of the cloth when the
  paper is lit, compared at G=32 vs G=64 in this container (G=64 vs G=128 on the GPU machine); report the numbers.
- ADR-0002 (MAC storage, accessor keeps the public `vel` shape) and ADR-0003 (energy split, re-radiation, clamps)
  in the Appendix B format, status accepted, with PRD change-log entries appended to `docs/PRD.md`.

**Acceptance:** all non-slow tests pass; the slow long-run test passes once (`make test-slow`, wall time reported).

## Task 9: P0-009 Volume renderer and two-view compositor

**Files:** `src/livefire/render/volume.py` (ray march kernel), `src/livefire/render/blackbody.py` (LUT),
`src/livefire/render/overlay.py` (char/removed splat into per-view buffers), `src/livefire/render/compositor.py`
(`Compositor` implementing `RendererBackend`), `configs/render.yaml`, `tests/test_render.py`.

- Per view, per pixel of a `volume_scale` resolution buffer (default 0.5): ray from the camera centre through the
  pixel centre in world space (use geometry helpers; K scaled for the reduced resolution), intersect the ROI AABB,
  march from `t_near` to `min(t_far, t_occ)` where `t_occ` = `view_depth` converted to ray distance
  (`zdepth_to_raydist`) minus a feather band `feather_m`; hole pixels march nothing. Fixed step `step_factor * h`,
  early exit when transmittance < `t_min`. Sampling of T_g and soot is trilinear in world space.
- Emission: blackbody LUT (Planck spectrum integrated against CIE 1931 2-degree CMFs, converted to linear sRGB, each
  colour normalised to unit luminance), 64 entries from 800 K to 3000 K; radiance
  `L = emission_scale * lut(T) * clamp((T - T_emit_min) / (T_emit_ref - T_emit_min), 0, inf) ^ emission_power`.
  Soot: absorption `sigma_a = k_abs * soot`, single-scatter ambient term `soot_albedo * ambient`. Emission is
  absorbed by soot along the ray. All noise (none in P0) would be world-space: no screen-space effects.
- Overlay: splat `char` and `removed` from elements into each view (disc splat like Task 6, accepted only where
  `|z_elem - view_depth| < depth_tol`); composite base = `frame_linear * (1 - char_dark * char)` then mix towards
  `ash_color` by `removed`.
- Composite: sRGB -> linear, base overlay, `out = base * T_vol + L_vol`, fixed exposure `exposure`, simple
  tone curve `x / (1 + x)` applied only to the added light? Use: `out = base * T + tonemap(L)` where
  `tonemap(L) = L / (1 + L)`; clamp; linear -> sRGB uint8. Upsample the volume buffers (T, L) bilinearly to full res.
- `configs/render.yaml`: `volume_scale 0.5`, `step_factor 0.5`, `t_min 0.01`, `feather_m 0.01 m`, `emission_scale 4.0`,
  `T_emit_min 700 K`, `T_emit_ref 1400 K`, `emission_power 2.0`, `k_abs 2000 m^2/kg`, `soot_albedo 0.2`,
  `ambient 0.1`, `char_dark 0.85`, `ash_color [0.12, 0.11, 0.10]`, `depth_tol 0.02 m`, `exposure 1.0`; all artistic,
  `{value, unit, range, status}` form (lists allowed as value).
- Tests: **cross-view consistency (M7 analogue on the synthetic rig):** for 5 seeded 3D points above the table that
  are visible from both cameras, set a Gaussian hot blob (T_g peak 1600 K, sigma 2 h) at the point on an otherwise
  T_amb field with no soot, render each view at volume_scale 1.0, compute the luminance-weighted centroid of `L_vol`
  and compare with `project(calib, point)`: median error < 6 px at 720p (report the actual median and max);
  **occlusion:** a blob placed just behind the wood block as seen from cam0 (choose the point by casting the cam0
  ray through the block centre and going 0.1 m past the far face) while visible from cam1: cam0 luminance inside a
  15 px radius of its projection is < 5 % of cam1's peak region sum; holes: forcing a hole mask suppresses emission;
  blackbody LUT: 1000 K is reddish (R > G > B), 3000 K has B/R higher than 1000 K; compositing round trip: no fire,
  no char -> output equals input frame within 1 LSB.

**Acceptance:** tests pass; `python -m livefire.render.compositor --demo` writes two composited PNGs with a
prescribed blob to `out/`.

## Task 10: P0-010 App: lifecycle, scheduler, HUD, record and replay

**Files:** `src/livefire/app/lifecycle.py`, `src/livefire/app/runner.py`, `src/livefire/app/hud.py`,
`src/livefire/app/replay.py`, `src/livefire/app/__main__.py`, `tests/test_lifecycle.py`, `tests/test_replay.py`.

- Lifecycle `UNINITIALIZED -> CALIBRATING -> ACQUIRING -> READY -> SIMULATING <-> PAUSED`, plus `DEGRADED` (a flag
  shown alongside SIMULATING) and `ERROR`; explicit allowed-transition table; illegal transitions raise.
  In P0, CALIBRATING loads the synthetic calibration and ACQUIRING renders the synthetic views and builds the
  snapshot.
- Runner (headless): two `SyntheticCameraSource`s (capture threads, newest-frame-wins), `FireSimulation`,
  `Compositor`. Display loop: take the newest frame per camera, `sim.run_frame(wall_dt)`, render both views,
  draw HUD, write to an MP4 (OpenCV `VideoWriter`, mp4v) side by side, and log per-frame timings (`StageTimer`) plus
  `t_present - t_arrive` latency and frame age. HUD text: state, delivered fps, p95 frame ms, sim rt factor, sim
  time, degradation level, hole fraction per view, element count, "SYNTHETIC SCENE" label. Degradation ladder (PRD 5.6)
  as a data structure with levels; P0 implements level 1 (lower `volume_scale`) and level 2 (larger `step_factor`),
  triggered when the rt factor < 0.9 for 30 frames, shown in HUD and logged (never silent).
- Events: scripted ignition `--ignite "x,y,z@t"` (sim time), material override `--override obj_id=material@t`.
  Record: `out/<run>/events.jsonl` (sim step index + event), `config_snapshot.yaml`, `seed`, final
  `FireStateHost.digest()`. Replay: `python -m livefire.app replay out/<run>` reruns the sim only (no rendering
  unless `--render`), applying events at their recorded step indices, and compares the digest (CPU: exact;
  report mismatch as failure).
- CLI: `python -m livefire.app run --scene table_v1 --seconds 8 --ignite "-0.075,0.0,0.05@0.5" --out out/demo`
  (`make run` uses this). On CPU this may run far below real time: the sim stays fixed-dt and the rt factor
  reports it honestly; offer `--offline` mode that steps the sim exactly `dt_sim * fps`-matched per output frame
  for making a reference video, labelled "OFFLINE (not real-time)" in the HUD.
- Tests: transition table (legal and illegal); runner smoke test (headless, 10 frames, tiny G=32, files written);
  replay digest equality for a recorded run with an ignition and an override; override changes behaviour (metal
  override on paper -> paper never burns).

**Acceptance:** tests pass; `make run` produces `out/demo/composite.mp4`.

## Task 11: P0-011 Benchmark harness and baseline CSV

**Files:** `bench/run_bench.py`, `bench/workloads.yaml`, `bench/plots.py`, `src/livefire/common/bench_csv.py`
(`BenchmarkRecorder` implementation), `tests/test_bench_csv.py`, `bench/results/baseline_cpu_<date>.csv`,
`bench/plots/*.png`.

- `BenchmarkRecorder` writes rows with exactly the PRD Appendix C columns, in order; fills environment fields
  automatically (git sha, date, OS, python, framework version, gpu/driver/cuda via `nvidia-smi` if present else
  `none`, power_mode `unknown` unless provided). VRAM fields from framework counters when on GPU, empty on CPU.
- Workloads (`workload_version: 1`): `sim_step` coupled at G in {64, 96, 128} on table_v1 with paper burning
  (pre-roll 120 steps after ignition so fire exists), `surface_step`, `gas_step`, `projection`, `render_view`
  (one view at 720p, volume_scale 0.5 and 1.0), `frame_two_views` (sim run_frame + 2 renders + composite).
  Rules PRD 11.4: warm-up discarded, >= 5 repeats, median and p95 (and p99), `samples` = total timed samples.
  `--quick` flag for CI-size runs. `make bench` runs the full set with the device from app.yaml.
- `bench/plots.py` reads only CSVs and writes scaling plots (stage ms vs G) to `bench/plots/`; `make plots`.
- Test: schema/column order, a row round-trip, required fields non-empty.
- Run the baseline here (CPU) and commit the CSV and plots, labelled CPU. If the full G=128 set exceeds 30 minutes
  on CPU, reduce samples (not the workload) and record that in `notes`.

**Acceptance:** CSV committed with all workloads; plots generated from it.

## Task 12: P0-012 Documentation stubs

**Files:** `docs/ARCHITECTURE.md`, `docs/NUMERICAL_METHODS.md`, `docs/COORDINATE_CONVENTIONS.md`,
`docs/TEST_STRATEGY.md`, `docs/PERFORMANCE_BUDGET.md`, `README.md` (update).

- Each file points to the PRD section it implements (no duplicated PRD text) and records only what the code adds:
  ARCHITECTURE: module map (actual files), lifecycle, ownership (snapshot host read-only, device mirrors owned by
  backends), and the PRD 6.2 per-field table: units, memory layout (actual array shapes, MAC faces), owner,
  initialisation. NUMERICAL_METHODS: list ADR-0002/0003 and the controller rulings (edge-flux conduction, sampling
  offset, sub-stepping), parameter files. COORDINATE_CONVENTIONS: pointer to PRD 4 plus the pixel-centre convention
  and `look_at` construction. TEST_STRATEGY: test files mapped to PRD 10 items and markers. PERFORMANCE_BUDGET:
  pointer to PRD 5.4/11, current CPU baseline CSV path, VRAM table "pending GPU run".
- README: setup, commands, CPU-only note.

**Acceptance:** links resolve; no PRD paragraphs copied.

## Task 13: P0-013 Literature review and novelty verdict

**Files:** `docs/literature.md`.

- Time-boxed web search (PRD section 1): live, real-time, streaming, multi-view, camera-based, AR, depth-based,
  fire, smoke, combustion, material-aware. Verify the three cited works (FieryGS arXiv 2605.00177, WildFireGS arXiv
  2608.11100, Wong et al. 2025 Fire 8(4):132) actually exist and say what they do; mark anything unverifiable
  "unverified". Find newer related work (2024-2026). For each entry: citation, URL, what it does, live or offline,
  views, material-aware or not, relation to the thesis.
- Verdict section: does a live, two-view, material-aware fire previs system already exist? If yes, the narrowed
  thesis. No fabricated citations: every entry has a URL you actually opened.

**Acceptance:** file committed; every citation has a URL.

## Task 14: P0-014 Gate 0 report

**Files:** `docs/reports/phase-0.md`, `docs/STATUS.md`.

- Written by the controller from committed artifacts only: tasks done, test results (`make test`, `make test-slow`
  output summaries), benchmark CSV paths and numbers quoted from them, ADR list, limitations, deviations, Gate 0
  outcome per PRD 0.4 against each Gate 0 criterion.
