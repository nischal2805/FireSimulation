# Coordinate conventions

Spec: `docs/PRD.md` section 4 (geometry contract, mandatory for every module). Read it there; this file adds only
what the code does with it. Implementation: `src/livefire/common/geometry.py`, tested in `tests/test_geometry.py`.

## Pixel centres

Integer pixel coordinates `(u, v)` denote pixel centres (OpenCV convention). `pixel_grid(width, height)` returns
float64 grids `u, v` of shape `(height, width)` built from `arange`, so `(0, 0)` is the centre of the top-left pixel.
The intrinsics in `configs/camera.yaml` follow the same rule. For the synthetic rig at 1280 x 720, `cx = 639.5` and
`cy = 359.5` are the centres of the image, not `W/2` and `H/2`.

## Transforms and back-projection

- Every transform is named `T_<to>_from_<from>`. Camera poses are `T_world_from_camera` (4x4, camera to world).
  `invert_transform` and `transform_points` take `T_b_from_a` and act on points in frame `a`.
- `make_K(fx, fy, cx, cy)` builds the 3x3 pixel intrinsic matrix for the stored resolution.
- Back-projection of a depth pixel: `p_c = d * inv(K) @ [u, v, 1]^T`, then `p_w = T_world_from_camera @ [p_c, 1]^T`.
  `backproject(calib, uv, z)` and `backproject_depth(calib, depth, valid)` implement this for one pixel and for a
  whole image. `d` is Z-depth along camera +Z in metres, not ray distance. `zdepth_to_raydist` and
  `raydist_to_zdepth` convert between the two.
- Invalid depth pixels hold `0.0` with `valid = False`. They are never NaN.

## `look_at` construction

`look_at(position, target, up=(0, 0, 1))` returns `T_world_from_camera` with OpenCV axes:

- camera `+Z` (forward) is the normalised `target - position`;
- camera `+X` (right) is `normalise(forward x up)`;
- camera `+Y` (down) is `forward x right`, so the three columns are right-handed.

The translation column is `position`. When the view direction is parallel to `up` (a straight-down camera),
the cross product degenerates. The code then uses world `+Y` as the up vector instead of raising.

## Voxel indexing

Voxel centres are `roi_min + (i + 0.5) * h`, with `h = roi_size / N`. Use only the accessors in
`livefire.common.geometry` (`VoxelGrid.h`, `centers`, `index_of`, `linear_index`, `unravel`, `in_bounds`) for index
and centre arithmetic. Do not repeat the formula inline.

Gas grid faces use the staggered layout in `docs/ARCHITECTURE.md` (ADR-0002).
