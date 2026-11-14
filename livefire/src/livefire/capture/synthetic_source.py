"""Synthetic camera source: re-emits one pre-rendered frame at a fixed fps (newest-wins)."""

from __future__ import annotations

import threading
import time

import numpy as np

from livefire.capture.types import Frame


class SyntheticCameraSource:
    """Implements ``CameraSource``. A background thread publishes the static ``rgb`` (uint8
    (H,W,3) sRGB, copied and made read-only) every 1/fps seconds with ``t_arrive_ns =
    perf_counter_ns()`` and increasing ``seq``; only the newest frame is kept. Thread-safe."""

    def __init__(self, cam_id: int, rgb: np.ndarray, calib_version: int, fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        img = np.ascontiguousarray(rgb, dtype=np.uint8).copy()
        img.flags.writeable = False
        self._cam_id, self._rgb, self._calib_version, self._period = cam_id, img, calib_version, 1.0 / fps
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def cam_id(self) -> int:
        return self._cam_id

    def _publish(self) -> None:
        f = Frame(self._cam_id, self._rgb, time.perf_counter_ns(), self._seq, self._calib_version)
        self._seq += 1
        with self._lock:
            self._latest = f

    def _run(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            self._publish()
            nxt += self._period
            self._stop.wait(max(0.0, nxt - time.perf_counter()))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._publish()  # first frame available immediately
        self._thread = threading.Thread(target=self._run, name=f"synthcam{self._cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest
