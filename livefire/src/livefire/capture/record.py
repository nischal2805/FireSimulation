"""Multi-camera recorder: per-camera MP4 + ``timestamps.csv`` + ``manifest.yaml`` (with sha256 of files)."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

import cv2
import yaml

from livefire.capture.types import Frame


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Recorder:
    """Writes each distinct ``Frame`` (per ``cam_id``, by ``seq``) to ``<out_dir>/cam<id>.mp4`` (mp4v) and a
    row ``cam_id,seq,t_arrive_ns,frame_index`` to ``<out_dir>/timestamps.csv``. ``sources_info`` maps
    cam_id -> description recorded in the manifest (source, requested fps). ``close()`` finalises the MP4s
    and writes ``manifest.yaml`` (source, resolution, fps, frame count, sha256 of every file, date) and
    returns the manifest dict. The MP4 fps is the declared ``fps`` per camera (timestamps.csv holds the
    true arrival times). Not thread-safe (single owner)."""

    def __init__(self, out_dir: str | Path, sources_info: dict[int, dict[str, Any]], fps: dict[int, float]) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._info, self._fps = sources_info, fps
        self._writers: dict[int, cv2.VideoWriter] = {}
        self._size: dict[int, tuple[int, int]] = {}
        self._count: dict[int, int] = {}
        self._last_seq: dict[int, int] = {}
        self._csv_f = (self.dir / "timestamps.csv").open("w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["cam_id", "seq", "t_arrive_ns", "frame_index"])

    def write(self, frame: Frame) -> bool:
        """Write ``frame`` unless its seq was already written for that camera; returns whether written."""
        c = frame.cam_id
        if self._last_seq.get(c, -1) >= frame.seq:
            return False
        h, w = frame.rgb.shape[:2]
        if c not in self._writers:
            vw = cv2.VideoWriter(str(self.dir / f"cam{c}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self._fps.get(c, 30.0) or 30.0, (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"cannot open MP4 writer for cam {c}")
            self._writers[c], self._size[c], self._count[c] = vw, (w, h), 0
        if self._size[c] != (w, h):
            raise ValueError(f"cam {c}: frame size changed {self._size[c]} -> {(w, h)}")
        self._writers[c].write(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        self._csv.writerow([c, frame.seq, frame.t_arrive_ns, self._count[c]])
        self._count[c] += 1
        self._last_seq[c] = frame.seq
        return True

    def close(self) -> dict[str, Any]:
        for vw in self._writers.values():
            vw.release()
        self._csv_f.close()
        cams = []
        for c in sorted(self._writers):
            f = self.dir / f"cam{c}.mp4"
            cams.append({
                "cam_id": c, **self._info.get(c, {}), "resolution": list(self._size[c]),
                "fps": float(self._fps.get(c, 30.0)), "frames": self._count[c],
                "file": f.name, "sha256": sha256_file(f),
            })  # fmt: skip
        ts = self.dir / "timestamps.csv"
        manifest = {
            "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "cameras": cams,
            "timestamps": {"file": ts.name, "sha256": sha256_file(ts)},
        }
        (self.dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        return manifest
