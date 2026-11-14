"""``VideoFileSource``: play a recorded clip as a camera (tests, replay, benchmarks)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from livefire.capture.opencv_source import _ThreadedSource


class VideoFileSource(_ThreadedSource):
    """Implements ``CameraSource`` over a video file. Output as ``OpenCVCameraSource`` (RGB uint8 (H,W,3),
    read-only, newest wins, ``t_arrive_ns`` stamped at publish). ``realtime=True`` paces frames at the file's
    fps (or ``fps`` if given); ``realtime=False`` reads as fast as possible. ``loop=True`` restarts at EOF
    (``seq`` keeps increasing); otherwise the status becomes ``ended`` and the last frame stays available.
    ``start`` raises ``RuntimeError`` if the file cannot be opened. ``calib_version`` is stamped on frames."""

    def __init__(
        self,
        cam_id: int,
        path: str | Path,
        realtime: bool = True,
        loop: bool = False,
        fps: float = 0.0,
        calib_version: int = 0,
    ) -> None:
        super().__init__(cam_id, calib_version)
        self._path, self._realtime, self._loop, self._fps = Path(path), realtime, loop, float(fps)
        self.actual_width = self.actual_height = 0
        self.actual_fps = 0.0
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            self._status = "failed"
            raise RuntimeError(f"camera {self._cam_id}: cannot open video file {self._path}")
        self._cap = cap
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.actual_fps = self._fps if self._fps > 0 else (file_fps if file_fps > 0 else 30.0)
        self._status = "connected"
        self._spawn(self._run)

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None
        period = 1.0 / self.actual_fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                if not self._loop or cap.get(cv2.CAP_PROP_POS_FRAMES) == 0:
                    self._status = "ended"
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self._publish_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if self._realtime:
                nxt += period
                if self._stop.wait(max(0.0, nxt - time.perf_counter())):
                    break
            else:
                time.sleep(0)
        cap.release()
        self._cap = None
