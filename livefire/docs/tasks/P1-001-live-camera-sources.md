## Task 1: P1-001 Live camera sources and capture tool

**Files:** `src/livefire/capture/opencv_source.py`, `src/livefire/capture/file_source.py`,
`src/livefire/capture/factory.py`, `src/livefire/capture/record.py`, `configs/cameras.yaml`,
`src/livefire/app/capture_cli.py` (+ wiring in `app/__main__.py`), `pyproject.toml` (opencv-python),
`tests/test_capture_live.py`, Makefile + README rows.

- `OpenCVCameraSource(CameraSource)`: source = int (USB index) or str (URL: MJPEG over HTTP such as the Android
  "IP Webcam" app `http://<phone-ip>:8080/video`, DroidCam `http://<ip>:4747/video`, or RTSP). Background thread
  grabs continuously (`grab()` loop, `retrieve()` only for the newest), keeps only the newest frame, stamps
  `t_arrive_ns = perf_counter_ns()` on arrival, increments `seq`, tracks measured fps and dropped/stale counts.
  Requested width/height/fps from config (best effort; report actual). On read failure: reconnect with exponential
  backoff (0.5 s .. 5 s), status `connected/reconnecting/failed` exposed. Windows: prefer `cv2.CAP_DSHOW` for USB
  indexes (config `backend: auto|dshow|msmf|v4l2|ffmpeg`). Frames are BGR from OpenCV: convert to the RGB `Frame`
  contract. `calib_version` attached from the active calibration (0 if uncalibrated).
- `VideoFileSource(CameraSource)`: plays a recorded clip as a camera, real-time paced (default) or as-fast-as-read,
  loop option; same newest-frame-wins semantics; used for tests, replay and benchmarks.
- `make_source(cfg) -> CameraSource` factory for `type: usb|url|file|synthetic`.
- `configs/cameras.yaml`: list of cameras `{cam_id, type, source, width, height, fps, backend}`; ship an example with
  cam0 = USB index 0 and cam1 = phone URL placeholder `http://192.168.1.50:8080/video`.
- Capture tool `python -m livefire.app capture [--record DIR --seconds N]`: opens a window with both feeds side by
  side, overlays per-camera measured fps, frame age (ms since t_arrive), inter-camera skew (ms between newest
  t_arrive values), connection status; `--headless` prints a summary instead. `--record` writes per-camera MP4 +
  `timestamps.csv` (seq, t_arrive_ns) + `manifest.yaml` (source, resolution, fps, sha256 of files, date).
- Glass-to-glass helper: `python -m livefire.app latency-timer` shows a large ms counter window to film with a
  camera; `capture --show-timer` overlays the live counter next to the camera feed so a screenshot shows both values
  (difference = glass-to-glass latency, PRD 5.5). Document the procedure in the README.
- Tests: VideoFileSource on a small generated MP4 (seq increases, newest-frame-wins, loop); OpenCVCameraSource
  against a local MJPEG HTTP server started in a thread (`http.server`, multipart/x-mixed-replace of generated JPEG
  frames with frame numbers encoded as pixel values): frames arrive, newest wins (consumer reading slower than
  producer sees skipped seq), reconnect after the server is stopped and restarted, status transitions; RGB order
  verified; recorder writes files + manifest with correct hashes.

**Acceptance:** tests pass headless; `capture --headless --seconds 2` on a file source prints fps/skew.

