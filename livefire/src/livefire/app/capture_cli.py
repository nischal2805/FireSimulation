"""Capture tool (``python -m livefire.app capture``) and glass-to-glass ``latency-timer`` (PRD 5.5).

Pure, tested logic (``camera_stats``, ``compose_view``, ``timer_ms``, ``run_capture``) with a thin
``cv2.imshow`` layer on top. Windows-safe: pathlib only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from livefire.capture.factory import load_camera_configs, make_source
from livefire.capture.record import Recorder
from livefire.common.interfaces import CameraSource

TIMER_WRAP_MS = 100_000


def timer_ms() -> int:
    """Wall-clock milliseconds modulo 100 000. Wall clock (not perf_counter) so two processes on one
    machine (the latency-timer window and the capture overlay) show comparable values."""
    return int(time.time() * 1000) % TIMER_WRAP_MS


def camera_stats(sources: list[CameraSource], now_ns: int | None = None) -> dict[str, Any]:
    """Per-camera ``{cam_id, fps, age_ms, status, seq}`` plus inter-camera ``skew_ms`` (absolute difference of
    the newest ``t_arrive_ns`` of the first two cameras with frames; None otherwise)."""
    now = time.perf_counter_ns() if now_ns is None else now_ns
    cams, arrivals = [], []
    for s in sources:
        f = s.latest()
        fps = getattr(s, "measured_fps", lambda: 0.0)()
        status = getattr(s, "status", "connected")
        cams.append({
            "cam_id": s.cam_id, "fps": fps, "status": status,
            "age_ms": None if f is None else (now - f.t_arrive_ns) / 1e6,
            "seq": None if f is None else f.seq,
        })  # fmt: skip
        if f is not None:
            arrivals.append(f.t_arrive_ns)
    skew = abs(arrivals[0] - arrivals[1]) / 1e6 if len(arrivals) >= 2 else None
    return {"cameras": cams, "skew_ms": skew}


def _label(img: np.ndarray, lines: list[str], org: tuple[int, int] = (8, 22)) -> None:
    for i, t in enumerate(lines):
        p = (org[0], org[1] + 22 * i)
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def compose_view(sources: list[CameraSource], stats: dict[str, Any], height: int = 360, show_timer: bool = False) -> np.ndarray:
    """BGR image (for ``imshow``): feeds side by side scaled to ``height`` px, each labelled with fps, frame
    age (ms), status; skew in the top-left of the first panel; optional live ms counter panel at the right."""
    panels = []
    for s, st in zip(sources, stats["cameras"], strict=True):
        f = s.latest()
        if f is None:
            panel = np.zeros((height, int(height * 16 / 9), 3), np.uint8)
        else:
            h, w = f.rgb.shape[:2]
            panel = cv2.resize(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), (max(1, round(w * height / h)), height))
        age = "n/a" if st["age_ms"] is None else f"{st['age_ms']:.0f} ms"
        lines = [f"cam{st['cam_id']} {st['status']}", f"fps {st['fps']:.1f}  age {age}"]
        if panels == [] and stats["skew_ms"] is not None:
            lines.append(f"skew {stats['skew_ms']:.1f} ms")
        _label(panel, lines)
        panels.append(panel)
    if show_timer:
        panel = np.zeros((height, height, 3), np.uint8)
        cv2.putText(panel, f"{timer_ms():05d}", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        _label(panel, ["live timer (ms)"])
        panels.append(panel)
    return np.hstack(panels)


def summary_line(stats: dict[str, Any]) -> str:
    parts = []
    for c in stats["cameras"]:
        age = "n/a" if c["age_ms"] is None else f"{c['age_ms']:.0f}ms"
        parts.append(f"cam{c['cam_id']} {c['status']} fps={c['fps']:.1f} age={age}")
    skew = stats["skew_ms"]
    parts.append("skew=" + ("n/a" if skew is None else f"{skew:.1f}ms"))
    return "  ".join(parts)


def run_capture(
    sources: list[CameraSource],
    seconds: float | None,
    headless: bool,
    record_dir: Path | None = None,
    show_timer: bool = False,
    declared_fps: dict[int, float] | None = None,
    source_info: dict[int, dict[str, Any]] | None = None,
    print_fn: Any = print,
) -> dict[str, Any]:
    """Start ``sources``, run for ``seconds`` (None = until 'q'/Esc, windowed only), optionally record, stop.
    Returns ``{"stats": last camera_stats, "manifest": dict | None}``. Headless prints a one-line status each
    second and a final summary. Sources are always stopped."""
    rec = Recorder(record_dir, source_info or {}, declared_fps or {}) if record_dir is not None else None
    t0, last_print, stats = time.perf_counter(), 0.0, {"cameras": [], "skew_ms": None}
    try:
        for s in sources:
            s.start()
        t0 = time.perf_counter()
        while seconds is None or time.perf_counter() - t0 < seconds:
            if rec is not None:
                for s in sources:
                    f = s.latest()
                    if f is not None:
                        rec.write(f)
            stats = camera_stats(sources)
            now = time.perf_counter() - t0
            if headless:
                if now - last_print >= 1.0:
                    print_fn(f"[{now:4.1f}s] {summary_line(stats)}")
                    last_print = now
                time.sleep(0.005)
            else:
                cv2.imshow("livefire capture", compose_view(sources, stats, show_timer=show_timer))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        for s in sources:
            s.stop()
        if not headless:
            cv2.destroyAllWindows()
    manifest = rec.close() if rec is not None else None
    print_fn(f"summary: {summary_line(stats)}")
    return {"stats": stats, "manifest": manifest}


def run_latency_timer() -> None:
    """Window with a large wall-clock ms counter (modulo 100 000) to film with a camera. Esc/q quits."""
    while True:
        img = np.zeros((300, 900, 3), np.uint8)
        cv2.putText(img, f"{timer_ms():05d}", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 10, cv2.LINE_AA)
        cv2.imshow("livefire latency-timer (ms)", img)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def add_subparsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    c = sub.add_parser("capture", help="show/record live cameras from configs/cameras.yaml with fps/age/skew overlays")
    c.add_argument("--config", default=None, help="cameras yaml (default configs/cameras.yaml)")
    c.add_argument("--seconds", type=float, default=None, help="stop after N s (required with --headless; default 10)")
    c.add_argument("--headless", action="store_true", help="no window: print a status line per second and a summary")
    c.add_argument("--record", default=None, metavar="DIR", help="write per-camera MP4, timestamps.csv, manifest.yaml")
    c.add_argument("--show-timer", action="store_true", help="add a live ms counter panel (see latency-timer)")
    sub.add_parser("latency-timer", help="large ms counter window to film with a camera (glass-to-glass latency)")


def run_command(a: argparse.Namespace) -> int:
    if a.cmd == "latency-timer":
        run_latency_timer()
        return 0
    cfgs = load_camera_configs(a.config)
    sources = [make_source(c) for c in cfgs]
    seconds = a.seconds if a.seconds is not None else (10.0 if a.headless or a.record else None)
    info = {c["cam_id"]: {"type": c["type"], "source": c["source"]} for c in cfgs}
    fps = {c["cam_id"]: float(c.get("fps", 30)) for c in cfgs}
    try:
        run_capture(sources, seconds, a.headless, Path(a.record) if a.record else None, a.show_timer, fps, info)
    except RuntimeError as e:
        print(f"capture failed: {e}", file=sys.stderr)
        return 1
    return 0
