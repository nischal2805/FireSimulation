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

