## Task 3: P1-003 ChArUco calibration with held-out validation

**Files:** `src/livefire/calib/{board.py,intrinsics.py,extrinsics.py,store.py,synthetic_board.py}`,
`src/livefire/app/calib_cli.py`, `configs/calibration.yaml`, `tests/test_calibration.py`, Makefile `calibrate`
target (replaces the P0 stub) + README rows.

- `configs/calibration.yaml`: board `squares_x 7, squares_y 5, square_length_m 0.035, marker_length_m 0.026,
  dictionary DICT_5X5_100` (fits A4 landscape at 100 % scale), `min_views 20`, `holdout_fraction 0.2`, seed.
  The user must measure the printed square size with a ruler and enter it (`square_length_m`), the marker length is
  scaled by the same factor.
- `board.py`: board object; `python -m livefire.app calib board --out board.png --dpi 300` writes a print-ready PNG
  (and states the physical size); detection helper returning ChArUco corners + ids per image.
- Intrinsics per camera: interactive capture `calib intrinsics --cam N` (window with detection overlay, coverage
  heat-map of corners over the image, auto-capture when the board is still and the pose differs from previous
  captures; `--from-dir` to use saved images). Fit with OpenCV (`calibrateCameraCharuco` or the 5.x equivalent),
  split views into fit/held-out with the seeded fraction, report RMS and median reprojection error on held-out views
  separately (PRD 7.1, target RMS < 1 px), save images + results.
- Extrinsics: board flat on the table visible from all cameras (`calib extrinsics`); per camera solvePnP on the
  ChArUco corners -> `T_board_from_camera`; convert to world per P1R2 (origin board centre, +Z up toward the
  cameras: flip if needed so cameras have z > 0); report per-camera reprojection RMS and the board-corner
  consistency between cameras (triangulate corners from both views, median distance to the known board layout).
- Store `data/calib/calib_vNNN.yaml` (+ images, manifest) with `calib_version` incremented, per camera K, dist,
  resolution, T_world_from_camera; `active` pointer file. Loaders produce `CameraCalibration` (P0) and undistortion
  maps (`cv2.initUndistortRectifyMap` with the same K as the undistorted K stored, alpha=0 policy documented).
- Synthetic board renderer (tests, MOCK-free ground truth): warp the board image into a camera with known K, dist,
  pose (homography for the plane + `cv2.projectPoints`-consistent distortion), add seeded noise/blur.
- Tests: intrinsics from 25 synthetic views recover fx, fy, cx, cy within 0.5 % and dist k1 within 0.02, held-out RMS
  < 0.5 px; extrinsics recover camera pose within 2 mm / 0.2 deg; world-frame convention: board centre maps to the
  origin, cameras above the table have z > 0, +Z toward cameras; version increments; round trip save/load; a
  calibration with a different resolution than the frames raises.

**Acceptance:** tests pass; the user-facing flow documented step by step in the README.

