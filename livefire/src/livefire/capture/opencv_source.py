"""OpenCV camera sources (USB index or URL) with a newest-frame-wins capture thread (PRD 5.3).

``_ThreadedSource`` holds the shared machinery (latest-frame slot, seq, fps/age/drop stats, status);
``OpenCVCameraSource`` adds device opening, reconnect with exponential backoff and BGR->RGB conversion.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from livefire.capture.types import Frame

BACKENDS = ("auto", "dshow", "msmf", "v4l2", "ffmpeg")
_FPS_WINDOW_S = 3.0


def resolve_backend(source: int | str, backend: str) -> int:
    """Map a config ``backend`` name to a ``cv2.CAP_*`` constant.

    ``auto``: USB index -> ``CAP_DSHOW`` on Windows, ``CAP_V4L2`` on Linux, else ``CAP_ANY``;
    URL -> ``CAP_FFMPEG``. Raises ``ValueError`` for an unknown name."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        if isinstance(source, str):
            return int(cv2.CAP_FFMPEG)
        if sys.platform == "win32":
            return int(cv2.CAP_DSHOW)
        return int(cv2.CAP_V4L2) if sys.platform.startswith("linux") else int(cv2.CAP_ANY)
    return int({"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2, "ffmpeg": cv2.CAP_FFMPEG}[backend])


class _ThreadedSource:
    """Shared base: a daemon thread publishes frames; only the newest is kept (thread-safe).

    Stats (all host-side, ``time.perf_counter_ns`` clock): ``measured_fps()`` over the last ~3 s of
    arrivals, ``frame_age_ms()`` of the newest frame, ``published`` (frames published), ``stale``
    (published frames replaced before any ``latest()`` call read them: producer faster than consumer),
    ``status``."""

    def __init__(self, cam_id: int, calib_version: int = 0) -> None:
        self._cam_id = cam_id
        self.calib_version = calib_version
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._unread = False
        self._seq = 0
        self._published = 0
        self._stale = 0
        self._arrivals: deque[int] = deque()
        self._status = "stopped"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def cam_id(self) -> int:
        return self._cam_id

    @property
    def status(self) -> str:
        return self._status

    @property
    def published(self) -> int:
        return self._published

    @property
    def stale(self) -> int:
        return self._stale

    def _publish_rgb(self, rgb: np.ndarray, t_arrive_ns: int | None = None) -> None:
        """Take ownership of ``rgb`` (uint8 (H,W,3), made read-only) and publish it with arrival time
        ``t_arrive_ns`` (default: now). Decoders pass the stamp taken right after ``grab()``."""
        t = time.perf_counter_ns() if t_arrive_ns is None else t_arrive_ns
        rgb.flags.writeable = False
        with self._lock:
            f = Frame(self._cam_id, rgb, t, self._seq, self.calib_version)
            self._seq += 1
            self._published += 1
            if self._unread:
                self._stale += 1
            self._unread = True
            self._latest = f
            self._arrivals.append(t)
            horizon = t - int(_FPS_WINDOW_S * 1e9)
            while self._arrivals and self._arrivals[0] < horizon:
                self._arrivals.popleft()

    def latest(self) -> Frame | None:
        with self._lock:
            self._unread = False
            return self._latest

    def measured_fps(self) -> float:
        """Arrival rate over the last ~3 s (0.0 with fewer than 2 arrivals or after a >3 s silence)."""
        with self._lock:
            arr = list(self._arrivals)
        now = time.perf_counter_ns()
        if len(arr) < 2 or now - arr[-1] > _FPS_WINDOW_S * 1e9:
            return 0.0
        return (len(arr) - 1) / ((arr[-1] - arr[0]) / 1e9)

    def frame_age_ms(self) -> float | None:
        """Milliseconds since the newest frame arrived (None before the first)."""
        with self._lock:
            f = self._latest
        return None if f is None else (time.perf_counter_ns() - f.t_arrive_ns) / 1e6

    join_timeout_s = 6.0

    def _alive(self) -> bool:
        """True if a capture thread is running. Raises ``RuntimeError`` if it is still winding down
        after a ``stop()`` whose join timed out (a restart must wait until it has exited)."""
        if self._thread is None or not self._thread.is_alive():
            return False
        if self._stop.is_set():
            raise RuntimeError(f"camera {self._cam_id}: previous capture thread is still stopping")
        return True

    def _spawn(self, target: Any) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=target, name=f"cam{self._cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread. If it does not exit within ``join_timeout_s`` (blocked in a read), the thread
        reference is kept, ``status`` is ``stopping`` and ``start()`` refuses until it has exited."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.join_timeout_s)
            if self._thread.is_alive():
                self._status = "stopping"
                return
            self._thread = None
        self._status = "stopped"


class OpenCVCameraSource(_ThreadedSource):
    """Implements ``CameraSource`` for a USB index (int) or a URL (MJPEG/RTSP, str) via ``cv2.VideoCapture``.

    Output ``Frame.rgb``: uint8 (H,W,3) RGB (converted from OpenCV BGR), host memory, read-only, newest wins.
    ``t_arrive_ns`` is stamped right after ``grab()`` returns True (before decode/colour conversion).
    ``width/height/fps`` (0 = leave default) are requested best effort; ``actual_width/height/fps`` report
    what the device says. ``fourcc`` (e.g. "MJPG", USB only) is set before width/height (often required for
    720p30 on Windows webcams).

    Retrieve only the newest: the thread ``grab()``s continuously; a grab arriving < 0.3 x the expected frame
    period after the previous one signals a backlog, so its ``retrieve()`` (decode) is skipped and counted in
    ``dropped``; a retrieve is forced if none happened for one period. Counters: ``grabbed``, ``retrieved``
    (= ``published``), ``dropped`` (grabbed but not decoded), ``stale`` (published but replaced unread).

    On read failure the device is released and reopened with exponential backoff
    ``backoff_min_s``..``backoff_max_s`` (0.5..5 s) forever. ``status``: ``connected``; ``reconnecting``;
    ``failed`` = disconnected for longer than ``failed_after_s`` (default 10 s) while still retrying;
    ``consecutive_failures`` counts failed reopens since the last good connection. ``start`` raises
    ``RuntimeError`` (status ``failed``) if the first open fails. Timeouts ``timeout_ms`` apply to URL
    sources only (USB opens without params; a failed open with params is retried without). ``calib_version``
    is stamped on every frame (0 = uncalibrated); this source does not undistort."""

    def __init__(
        self,
        cam_id: int,
        source: int | str,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        backend: str = "auto",
        calib_version: int = 0,
        backoff_min_s: float = 0.5,
        backoff_max_s: float = 5.0,
        failed_after_s: float = 10.0,
        timeout_ms: int = 3000,
        fourcc: str | None = None,
    ) -> None:
        super().__init__(cam_id, calib_version)
        if fourcc is not None and len(fourcc) != 4:
            raise ValueError(f"fourcc must be 4 characters, got {fourcc!r}")
        self._source, self._req, self.fourcc = source, (int(width), int(height), float(fps)), fourcc
        self._api = resolve_backend(source, backend)
        self._bmin, self._bmax, self._failed_after, self._timeout_ms = backoff_min_s, backoff_max_s, failed_after_s, timeout_ms
        self.join_timeout_s = timeout_ms / 1000.0 + 0.5
        self.actual_width = self.actual_height = 0
        self.actual_fps = 0.0
        self.reconnects = 0
        self.consecutive_failures = 0
        self.grabbed = self.retrieved = self.dropped = 0
        self._disconnected_since: float | None = None
        self._cap: cv2.VideoCapture | None = None

    @property
    def status(self) -> str:
        since = self._disconnected_since
        if self._status == "reconnecting" and since is not None and time.perf_counter() - since >= self._failed_after:
            return "failed"
        return self._status

    def _open(self) -> bool:
        cap = None
        try:
            if isinstance(self._source, str):
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_ms]
                try:
                    cap = cv2.VideoCapture(self._source, self._api, params)
                except cv2.error:
                    cap = None
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(self._source, self._api)
            else:
                cap = cv2.VideoCapture(self._source, self._api)
        except cv2.error:
            return False
        if not cap.isOpened():
            cap.release()
            return False
        w, h, fps = self._req
        if self.fourcc and not isinstance(self._source, str):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._cap = cap
        return True

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def start(self) -> None:
        if self._alive():
            return
        if not self._open():
            self._status = "failed"
            raise RuntimeError(f"camera {self._cam_id}: cannot open {self._source!r}")
        self._status, self._disconnected_since, self.consecutive_failures = "connected", None, 0
        self._spawn(self._run)

    def _period_s(self) -> float:
        fps = self._req[2] if self._req[2] > 0 else self.actual_fps
        return 1.0 / fps if fps > 1.0 else 1.0 / 30.0

    def _run(self) -> None:
        backoff = self._bmin
        last_grab = last_retrieve = 0
        while not self._stop.is_set():
            if self._cap is None:
                if self._stop.wait(backoff):
                    break
                if self._open():
                    self._status, self._disconnected_since = "connected", None
                    self.consecutive_failures, backoff = 0, self._bmin
                    self.reconnects += 1
                    last_grab = last_retrieve = 0
                else:
                    self.consecutive_failures += 1
                    backoff = min(backoff * 2.0, self._bmax)
                continue
            ok = self._cap.grab()
            t_arrive = time.perf_counter_ns()  # arrival stamp: before decode and colour conversion
            if not ok or self._stop.is_set():
                if not self._stop.is_set():
                    self._close()
                    self._status, self._disconnected_since = "reconnecting", time.perf_counter()
                continue
            self.grabbed += 1
            period_ns = self._period_s() * 1e9
            backlog = last_grab and (t_arrive - last_grab) < 0.3 * period_ns
            last_grab = t_arrive
            if backlog and (t_arrive - last_retrieve) < period_ns:
                self.dropped += 1
                continue
            ret, bgr = self._cap.retrieve()
            if not ret:
                self._close()
                self._status, self._disconnected_since = "reconnecting", time.perf_counter()
                continue
            last_retrieve = t_arrive
            self.retrieved += 1
            self._publish_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_arrive)
        self._close()
