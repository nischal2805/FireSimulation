## Task 2: P3-002 Object registry with cross-view association, integrated into acquisition

**Files:** `src/livefire/fusion/associate.py` (new), `src/livefire/scene/acquire.py` (optional kwargs),
`configs/acquisition.yaml` (`segment`/`assoc` blocks), `tests/test_associate.py` (new),
`tests/test_acquire_segmented.py` (new).

- `configs/acquisition.yaml` additions under `acquisition`: `assoc: {visibility_tol_m: {value: 0.02, unit: "m", range:
  [0.002, 0.1], status: artistic}, min_iou: {value: 0.3, unit: "-", range: [0.05, 0.95], status: artistic},
  iou_tie: 0.05, min_instance_surfels: 10}`.
- `associate.py`: `label_surfels(pos, cals, accumulated, segmentations, cfg) -> dict[int, NDArray[int32]]`
  (per-camera instance id per surfel, -1 when not visible / background, P3R3 visibility rule) and
  `associate_objects(per_cam_labels, geometric_labels, rgb_stats, cfg) -> AssociationResult` (`labels` (N,) world obj
  id, `observations: dict[obj, dict[cam, inst]]`, `categories: dict[obj, list[str]]`, `report`), implementing the
  union-find merge, IoU tie-break by CIELAB mean colour, table override rule and renumbering of P3R3.
- `acquire_with_report(..., segmenters: Mapping[int, SegmentationBackend] | SegmentationBackend | None = None)`:
  `None` keeps the P1 path exactly. Before editing, record a golden digest (sha256 of `obj_id`, `pos`, `material_id`
  bytes) of the existing two-camera synthetic acquisition fixture into the new test file as a constant, with the
  commit it was taken at; the test asserts `segmenters=None` still produces it. With segmenters: after accumulation,
  segment the P3R2 keyframe(s) per camera (stage `segment`), extract and cluster as today, then `label_surfels` +
  `associate_objects` replace the labels before `bin_surfels` (stage `associate`); categories go into
  `WorldObject.name`/`MaterialPrediction.category_hint`. This task adds only the `segmenters` kwarg;
  `material_backend` (Task 3) and `overrides` (Task 4) are added by those tasks, same pattern (default None = today).
- Report additions: per world object `observations`, `categories`, `source: segmentation|geometric`, unlabelled
  surfel fraction; timings `segment`, `associate`.
- Tests: `associate` unit tests on hand-built two-camera surfel sets: permuted ids merge correctly; an instance seen
  only in one camera becomes its own object; two instances overlapping one instance -> higher IoU wins, colour breaks a
  tie; invisible surfels never take an id. Integration (`tests/test_acquire_segmented.py`, same fixture style and noise
  as `tests/test_acquire.py`: Oracle depth + sigma 1 % + 3 % per-view scale; `OracleSegmentationBackend` with permuted
  ids): 4 objects + table; **cloth separated from the table** with > 90 % membership; every object's observations map
  to the permuted per-camera ids; single-camera acquisition with segmentation works; `segmenters=None` gives the same
  golden digest recorded before the change; timings include `segment` and `associate`.

**Acceptance:** fast tests green (slow ones by node id if any); `tests/test_acquire.py` unchanged and green.

