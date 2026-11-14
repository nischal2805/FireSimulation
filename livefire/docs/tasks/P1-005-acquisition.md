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

