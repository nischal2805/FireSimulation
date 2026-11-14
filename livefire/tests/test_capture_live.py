"""Live-camera sources + capture tool (P1-001). Headless: generated clips and a local MJPEG HTTP server."""

import hashlib
import http.server
import itertools
import socketserver
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from livefire.app.__main__ import main
from livefire.app.capture_cli import camera_stats, compose_view, run_capture, summary_line, timer_ms
from livefire.capture.factory import load_camera_configs, make_source
from livefire.capture.file_source import VideoFileSource
from livefire.capture.opencv_source import OpenCVCameraSource, resolve_backend
from livefire.capture.record import Recorder, sha256_file
from livefire.common.config import ConfigError
from livefire.common.interfaces import CameraSource

W, H = 64, 48


def _rgb(n: int) -> np.ndarray:
    """Test image: pure RGB (200, 100, 30) with the frame number in the top-left 8x8 block's red/green."""
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (200, 100, 30)
    img[:8, :8] = (n % 256, 255 - n % 256, 0)
    return img


def _wait(cond, timeout=8.0):
    t = time.time() + timeout
    while time.time() < t:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for n in range(12):
        vw.write(cv2.cvtColor(_rgb(n), cv2.COLOR_RGB2BGR))
    vw.release()
    return p


def test_file_source_seq_and_protocol(clip):
    s = VideoFileSource(3, clip, realtime=False, calib_version=7)
    assert isinstance(s, CameraSource)
    s.start()
    assert _wait(lambda: s.status == "ended")
    f = s.latest()
    s.stop()
    assert f.cam_id == 3 and f.calib_version == 7 and f.seq == 11 and s.published == 12  # all frames, in order
    assert f.rgb.shape == (H, W, 3) and not f.rgb.flags.writeable and f.rgb.flags.c_contiguous


def test_file_source_realtime_pacing(clip):
    s = VideoFileSource(0, clip, realtime=True, loop=True)
    s.start()
    time.sleep(0.5)
    assert s.latest() is not None
    n = s.published
    s.stop()
    assert 8 <= n <= 22  # ~30 fps for 0.5 s, not as-fast-as-possible
    assert s.status == "stopped"


def test_file_source_loop_and_newest_wins(clip):
    s = VideoFileSource(0, clip, realtime=False, loop=True)
    s.start()
    seqs = []
    for _ in range(5):
        time.sleep(0.05)
        seqs.append(s.latest().seq)
    s.stop()
    assert seqs == sorted(seqs) and seqs[-1] > 12  # looped past the 12-frame clip
    assert s.stale > 0  # producer far faster than the consumer: frames replaced unread
    assert seqs[1] - seqs[0] > 1  # skipped seqs, no queue


def test_file_source_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        VideoFileSource(0, tmp_path / "nope.mp4").start()


class _Mjpeg:
    """Local MJPEG server (multipart/x-mixed-replace) streaming generated JPEG frames; restartable on one port."""

    def __init__(self, fps: float = 30.0, port: int = 0):
        outer = self
        self.n = 0

        class H_(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not outer.stopping:
                        _ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(_rgb(outer.n), cv2.COLOR_RGB2BGR))
                        outer.n += 1
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg.tobytes() + b"\r\n")
                        time.sleep(1.0 / fps)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.stopping = False
        self.srv = Srv(("127.0.0.1", port), H_)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/video"

    def close(self):
        self.stopping = True
        self.srv.shutdown()
        self.srv.server_close()


def test_opencv_url_frames_rgb_and_stats():
    srv = _Mjpeg(fps=30)
    s = OpenCVCameraSource(1, srv.url, calib_version=2, backoff_min_s=0.1)
    try:
        s.start()
        assert s.status == "connected" and (s.actual_width, s.actual_height) == (W, H)
        assert _wait(lambda: s.published >= 15)
        f = s.latest()
        assert f.cam_id == 1 and f.calib_version == 2 and f.rgb.shape == (H, W, 3) and not f.rgb.flags.writeable
        # RGB order: background (200,100,30) is R-dominant; JPEG is lossy so allow tolerance.
        assert np.abs(f.rgb[30:, 30:].reshape(-1, 3).mean(0) - (200, 100, 30)).max() < 12
        assert 10 < s.measured_fps() < 40
        assert 0 <= s.frame_age_ms() < 1000
    finally:
        s.stop()
        srv.close()


def test_opencv_newest_wins_slow_consumer():
    srv = _Mjpeg(fps=60)
    s = OpenCVCameraSource(0, srv.url)
    try:
        s.start()
        assert _wait(lambda: s.latest() is not None)
        seqs = []
        for _ in range(4):
            time.sleep(0.25)
            seqs.append(s.latest().seq)
        assert all(b - a > 2 for a, b in itertools.pairwise(seqs))  # skipped frames, never queued
        assert s.stale > 0
    finally:
        s.stop()
        srv.close()


def test_opencv_reconnect_after_server_restart():
    srv = _Mjpeg(fps=30)
    port = srv.port
    s = OpenCVCameraSource(0, srv.url, backoff_min_s=0.2, backoff_max_s=0.5, timeout_ms=1500)
    srv2 = None
    try:
        s.start()
        assert _wait(lambda: s.published >= 5)
        srv.close()
        assert _wait(lambda: s.status == "reconnecting")
        n = s.published
        srv2 = _Mjpeg(fps=30, port=port)
        assert _wait(lambda: s.status == "connected" and s.published > n + 3, timeout=15)
        assert s.reconnects >= 1
    finally:
        s.stop()
        srv.close() if srv2 is None else srv2.close()


def test_opencv_start_failure_raises():
    s = OpenCVCameraSource(0, "http://127.0.0.1:1/video", timeout_ms=500)
    with pytest.raises(RuntimeError):
        s.start()
    assert s.status == "failed"
    s.stop()


def test_resolve_backend():
    import sys

    assert resolve_backend("http://x/video", "auto") == cv2.CAP_FFMPEG
    assert resolve_backend(0, "msmf") == cv2.CAP_MSMF
    expected = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    assert resolve_backend(0, "auto") == expected
    with pytest.raises(ValueError):
        resolve_backend(0, "nope")


def test_factory_and_shipped_config(clip):
    cfgs = load_camera_configs()
    assert [c["type"] for c in cfgs] == ["usb", "url"] and cfgs[1]["source"].startswith("http://")
    assert isinstance(make_source(cfgs[0]), OpenCVCameraSource)
    assert isinstance(make_source(cfgs[1]), OpenCVCameraSource)
    fs = make_source({"cam_id": 2, "type": "file", "source": str(clip), "realtime": False})
    assert isinstance(fs, VideoFileSource)
    syn = make_source({"cam_id": 0, "type": "synthetic", "fps": 20})
    syn.start()
    assert syn.latest().rgb.ndim == 3
    syn.stop()
    with pytest.raises(ConfigError):
        make_source({"cam_id": 0, "type": "bogus"})


def test_recorder_files_and_hashes(clip, tmp_path):
    out = tmp_path / "rec"
    a = VideoFileSource(0, clip, realtime=False)
    b = VideoFileSource(1, clip, realtime=False)
    a.start(), b.start()
    assert _wait(lambda: a.status == "ended" and b.status == "ended")
    rec = Recorder(out, {0: {"source": "A"}, 1: {"source": "B"}}, {0: 30.0, 1: 30.0})
    fa, fb = a.latest(), b.latest()
    assert rec.write(fa) and not rec.write(fa) and rec.write(fb)
    man = rec.close()
    for c in (0, 1):
        assert (out / f"cam{c}.mp4").stat().st_size > 0
    m = yaml.safe_load((out / "manifest.yaml").read_text())
    assert m == man or m["cameras"][0]["sha256"] == man["cameras"][0]["sha256"]
    for cam in m["cameras"]:
        assert cam["sha256"] == hashlib.sha256((out / cam["file"]).read_bytes()).hexdigest()
        assert cam["resolution"] == [W, H] and cam["frames"] == 1
    assert m["cameras"][0]["source"] == "A"
    assert m["timestamps"]["sha256"] == sha256_file(out / "timestamps.csv")
    rows = (out / "timestamps.csv").read_text().splitlines()
    assert rows[0] == "cam_id,seq,t_arrive_ns,frame_index" and len(rows) == 3


def test_camera_stats_skew_and_view(clip):
    a = VideoFileSource(0, clip, realtime=True, loop=True)
    b = VideoFileSource(1, clip, realtime=True, loop=True)
    a.start(), b.start()
    try:
        assert _wait(lambda: a.published > 3 and b.published > 3)
        st = camera_stats([a, b])
        assert st["skew_ms"] is not None and 0 <= st["skew_ms"] < 200
        assert all(c["fps"] > 0 and c["age_ms"] >= 0 and c["status"] == "connected" for c in st["cameras"])
        img = compose_view([a, b], st, height=96, show_timer=True)
        assert img.dtype == np.uint8 and img.shape[0] == 96 and img.shape[1] > 2 * 96
        assert "skew=" in summary_line(st)
    finally:
        a.stop(), b.stop()
    assert 0 <= timer_ms() < 100_000


def test_run_capture_headless_with_record(clip, tmp_path):
    lines = []
    srcs = [VideoFileSource(0, clip, loop=True), VideoFileSource(1, clip, loop=True)]
    res = run_capture(srcs, 1.0, True, tmp_path / "r", declared_fps={0: 30, 1: 30}, print_fn=lines.append)
    assert res["stats"]["skew_ms"] is not None
    assert len(res["manifest"]["cameras"]) == 2 and all(c["frames"] > 5 for c in res["manifest"]["cameras"])
    assert lines[-1].startswith("summary:") and "fps=" in lines[-1]
    assert all(s.status == "stopped" for s in srcs)


def test_cli_capture_headless_file_source(clip, tmp_path, capsys):
    cfg = tmp_path / "cams.yaml"
    cfg.write_text(yaml.safe_dump({"cameras": [{"cam_id": 0, "type": "file", "source": str(clip), "loop": True, "fps": 30},
                                               {"cam_id": 1, "type": "file", "source": str(clip), "loop": True}]}))
    assert main(["capture", "--headless", "--seconds", "1.2", "--config", str(cfg), "--record", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "fps=" in out and "skew=" in out and (tmp_path / "o" / "manifest.yaml").is_file()
