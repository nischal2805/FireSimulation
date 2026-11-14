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

