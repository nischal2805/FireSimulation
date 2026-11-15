"""``python -m livefire.app measure``: M4 geometry accuracy by tape measure (P1-007, PRD 11.1 M4).

You click two points on real objects in the frozen views, type the distance you measured with a tape, and the tool
stores the pair; ``eval/geometry_mae.py`` turns ``pairs.csv`` into the MAE. A click back-projects through the frozen
acquisition depth (``app.click.resolve_click``), exactly like click-to-ignite; clicks on holes are rejected. The two
points of a pair may be in different cameras (both are in the world frame). Same hole-free reading as in live mode.

* :class:`MeasureSession` is the pure, tested logic (clicks, typed digits, undo, CSV, report); :func:`render` draws the
  views and status lines headless; :func:`run_window` is the thin OpenCV loop. No camera or display is needed for tests.
* Keys in the window: left click picks a point; after two points type the tape distance in **cm**, Enter stores it
  (Backspace edits, Esc cancels the pair); ``u`` undoes the last pair, ``s`` saves, ``q`` or Esc quits (saves).
* Headless: ``--from-csv pixels.csv`` (columns ``cam_a,u_a,v_a,cam_b,u_b,v_b,tape_cm``) builds the pairs without a window.
* ``--acquire-only`` acquires, saves the bundle (``out/eval/acq_<time>/acquisition.npz``) and exits.

Output ``pairs.csv`` columns: ``id, cam_a, u_a, v_a, cam_b, u_b, v_b, x_a, y_a, z_a, x_b, y_b, z_b, d_recon_cm, d_tape_cm,
error_cm`` (``error_cm = d_recon_cm - d_tape_cm``; pixels are full-resolution undistorted view pixels; world metres,
+Z up, table plane z = 0).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from livefire.app.acq_bundle import AcqBundle, add_source_args, resolve_bundle
from livefire.app.click import resolve_click
from livefire.app.interactive import ViewLayout
from livefire.common.eval_csv import mae_report, read_csv, write_csv

PAIR_FIELDS = ("id", "cam_a", "u_a", "v_a", "cam_b", "u_b", "v_b", "x_a", "y_a", "z_a", "x_b", "y_b", "z_b",
               "d_recon_cm", "d_tape_cm", "error_cm")  # fmt: skip
WINDOW = "LiveFire measure"
MAX_TAPE_CM = 1000.0
_REASONS = {"hole": "no depth there (hole): click a nearby surface point", "outside": "outside the image",
            "no_depth": "no frozen depth for that camera"}  # fmt: skip


@dataclass(frozen=True)
class Point:
    """A resolved click: camera, full-resolution pixel and world point (m)."""

    cam: int
    u: float
    v: float
    world: np.ndarray


@dataclass(frozen=True)
class Pair:
    """Two points and the tape distance [cm]; ``d_recon_cm`` is the reconstructed distance."""

    a: Point
    b: Point
    tape_cm: float

    @property
    def d_recon_cm(self) -> float:
        return float(np.linalg.norm(self.a.world - self.b.world)) * 100.0

    @property
    def error_cm(self) -> float:
        return self.d_recon_cm - self.tape_cm

    def row(self, i: int) -> dict[str, Any]:
        return {"id": i, "cam_a": self.a.cam, "u_a": self.a.u, "v_a": self.a.v, "cam_b": self.b.cam, "u_b": self.b.u,
                "v_b": self.b.v, "x_a": self.a.world[0], "y_a": self.a.world[1], "z_a": self.a.world[2],
                "x_b": self.b.world[0], "y_b": self.b.world[1], "z_b": self.b.world[2], "d_recon_cm": self.d_recon_cm,
                "d_tape_cm": self.tape_cm, "error_cm": self.error_cm}  # fmt: skip


class MeasureSession:
    """State machine of the tape-measure tool over an acquisition bundle (pure; host NumPy).

    ``click(cam, u, v)`` takes a full-resolution view pixel; the first two accepted clicks form a pending pair that
    waits for a distance (``type_char`` / ``backspace`` / ``enter``, or ``submit(cm)`` directly). ``pairs`` are the stored
    pairs, ``message`` the status text, ``pending`` the 0-2 points of the open pair, ``typed`` the digits entered so far."""

    def __init__(self, bundle: AcqBundle) -> None:
        self.bundle = bundle
        self.snapshot = bundle.snapshot()
        self.pairs: list[Pair] = []
        self.pending: list[Point] = []
        self.typed = ""
        self.message = "click the first point"

    # ---- clicks and typing -----------------------------------------------------------------------------------------
    @property
    def awaiting_distance(self) -> bool:
        return len(self.pending) == 2

    def click(self, cam: int, u: float, v: float) -> bool:
        """Register a point; returns whether it was accepted (False for holes, outside, a pending pair waiting for its
        distance, or an unknown camera; ``message`` says why)."""
        if self.awaiting_distance:
            self.message = "type the tape distance in cm and press Enter (Esc cancels this pair)"
            return False
        r = resolve_click(self.snapshot, cam, u, v)
        if r.point is None:
            self.message = _REASONS.get(r.reason, r.reason)
            return False
        self.pending.append(Point(cam, float(u), float(v), r.point))
        if self.awaiting_distance:
            d = float(np.linalg.norm(self.pending[0].world - self.pending[1].world)) * 100.0
            self.message = f"reconstructed {d:.1f} cm: type the tape distance in cm, Enter to store"
        else:
            self.message = "click the second point"
        return True

    def type_char(self, ch: str) -> None:
        """Digit or ``.`` of the distance being typed (ignored unless two points are pending)."""
        if self.awaiting_distance and ch in "0123456789." and len(self.typed) < 8 and not (ch == "." and "." in self.typed):
            self.typed += ch

    def backspace(self) -> None:
        self.typed = self.typed[:-1]

    def enter(self) -> Pair | None:
        """Store the pair with the typed distance; None (and a message) if nothing valid is typed."""
        if not self.awaiting_distance:
            return None
        try:
            return self.submit(float(self.typed))
        except ValueError as e:
            self.message = f"bad distance {self.typed!r}: {e}"
            return None

    def submit(self, tape_cm: float) -> Pair:
        """Store the pending pair with ``tape_cm``. Raises ``ValueError`` for a non-positive / absurd distance or when no
        pair is pending."""
        if not self.awaiting_distance:
            raise ValueError("no pair pending: click two points first")
        if not (0.0 < tape_cm <= MAX_TAPE_CM) or not np.isfinite(tape_cm):
            raise ValueError(f"tape distance must be in (0, {MAX_TAPE_CM:.0f}] cm")
        p = Pair(self.pending[0], self.pending[1], float(tape_cm))
        self.pairs.append(p)
        self.pending, self.typed = [], ""
        self.message = (f"pair {len(self.pairs)}: reconstructed {p.d_recon_cm:.1f} cm vs tape {tape_cm:.1f} cm "
                        f"(error {p.error_cm:+.1f} cm). {self.summary_line()}")  # fmt: skip
        return p

    def cancel(self) -> None:
        """Drop the open (pending) points and typed digits."""
        self.pending, self.typed = [], ""
        self.message = "cancelled: click the first point"

    def undo(self) -> None:
        """Cancel an open pair, else remove the last stored pair."""
        if self.pending:
            self.cancel()
        elif self.pairs:
            self.pairs.pop()
            self.message = f"removed last pair ({len(self.pairs)} left)"
        else:
            self.message = "nothing to undo"

    # ---- results -------------------------------------------------------------------------------------------------
    def report(self) -> dict[str, float]:
        """M4 statistics of the stored pairs (``eval/geometry_mae.mae_report``); raises ``ValueError`` with no pairs."""
        if not self.pairs:
            raise ValueError("no pairs stored yet")
        return mae_report(np.array([p.error_cm for p in self.pairs]), np.array([p.tape_cm for p in self.pairs]))

    def summary_line(self) -> str:
        if not self.pairs:
            return "no pairs yet"
        r = self.report()
        return f"{len(self.pairs)} pairs, MAE {r['mae_cm']:.2f} cm"

    def save(self, path: str | Path) -> Path:
        """Write ``pairs.csv`` (``PAIR_FIELDS``); returns the path."""
        return write_csv(path, PAIR_FIELDS, [p.row(i + 1) for i, p in enumerate(self.pairs)])

    def load_pixel_pairs(self, path: str | Path) -> int:
        """Add pairs from a CSV ``cam_a,u_a,v_a,cam_b,u_b,v_b,tape_cm`` (headless mode). Returns how many were added;
        raises ``ValueError`` naming the first row that cannot be reconstructed."""
        n = 0
        for i, r in enumerate(read_csv(path), 1):
            self.cancel()
            ok = self.click(int(r["cam_a"]), float(r["u_a"]), float(r["v_a"])) and self.click(int(r["cam_b"]), float(r["u_b"]), float(r["v_b"]))
            if not ok:
                raise ValueError(f"{path} row {i}: {self.message}")
            self.submit(float(r["tape_cm"]))
            n += 1
        return n


# ---- drawing --------------------------------------------------------------------------------------------------
def layout_for(bundle: AcqBundle, max_w: int = 1600, max_h: int = 900) -> ViewLayout:
    cams = sorted(bundle.cameras)
    c0 = bundle.cameras[cams[0]]
    if any((bundle.cameras[c].width, bundle.cameras[c].height) != (c0.width, c0.height) for c in cams):
        raise ValueError("the measure window needs all cameras at the same resolution")
    return ViewLayout(c0.width, c0.height, len(cams), max_w, max_h)


def render(session: MeasureSession, layout: ViewLayout) -> np.ndarray:
    """BGR uint8 image for ``imshow``: the stored views side by side (cameras sorted), stored pairs in green with their
    ids, the pending points in yellow, and the status lines at the bottom. Raises ``ValueError`` if the bundle has no frames."""
    cams = sorted(session.bundle.cameras)
    if any(c not in session.bundle.rgb for c in cams):
        raise ValueError("the bundle has no camera frames to display")
    panes = [cv2.resize(np.ascontiguousarray(session.bundle.rgb[c][..., ::-1]), (layout.pane_w, layout.pane_h),
                        interpolation=cv2.INTER_AREA) for c in cams]
    img = np.concatenate(panes, axis=1)

    def mark(pt: Point, colour: tuple[int, int, int], text: str) -> tuple[int, int]:
        x, y = layout.from_view(cams.index(pt.cam), pt.u, pt.v)
        p = (round(x), round(y))
        cv2.circle(img, p, 6, colour, 2, cv2.LINE_AA)
        cv2.putText(img, text, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return p

    for i, pr in enumerate(session.pairs, 1):
        pa, pb = mark(pr.a, (0, 220, 0), f"{i}a"), mark(pr.b, (0, 220, 0), f"{i}b")
        if pr.a.cam == pr.b.cam:
            cv2.line(img, pa, pb, (0, 220, 0), 1, cv2.LINE_AA)
    for k, pt in enumerate(session.pending):
        mark(pt, (0, 230, 255), "ab"[k])
    for i, c in enumerate(cams):
        cv2.putText(img, f"cam{c}", (i * layout.pane_w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    bar = np.zeros((70, img.shape[1], 3), np.uint8)
    prompt = f"tape distance (cm): {session.typed}_" if session.awaiting_distance else session.summary_line()
    for k, t in enumerate([session.message[:160], prompt, "click 2 points, type cm + Enter | u undo | s save | q quit"]):
        cv2.putText(bar, t, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([img, bar], axis=0)


def handle_key(session: MeasureSession, code: int) -> str | None:
    """Map a ``cv2.waitKey`` code (already ``& 0xFF``; -1 = none) to session actions. Returns ``"quit"`` or ``"save"`` when
    the window loop must act, else None."""
    if code == -1:
        return None
    if session.awaiting_distance:
        if code in (13, 10):
            session.enter()
        elif code in (8, 127):
            session.backspace()
        elif code == 27:
            session.cancel()
        else:
            session.type_char(chr(code))
        return None
    ch = chr(code)
    if code == 27 or ch == "q":
        return "quit"
    if ch == "u":
        session.undo()
    elif ch == "s":
        return "save"
    return None


def run_window(session: MeasureSession, pairs_path: Path, max_w: int | None = None, max_h: int | None = None) -> None:
    """Interactive loop (thin OpenCV layer; not unit-tested because it needs a display)."""
    from livefire.app.window import fit_limits

    max_w, max_h = fit_limits(max_w, max_h)
    layout = layout_for(session.bundle, max_w, max_h - 80)
    cams = sorted(session.bundle.cameras)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, _: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = layout.to_view(x, y)
            if hit is not None:
                session.click(cams[hit[0]], hit[1], hit[2])

    cv2.setMouseCallback(WINDOW, on_mouse)
    saved = 0
    try:
        while True:
            cv2.imshow(WINDOW, render(session, layout))
            key = cv2.waitKey(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the X button
                break
            act = handle_key(session, -1 if key == -1 else key & 0xFF)
            if len(session.pairs) != saved or act == "save":
                session.save(pairs_path)  # autosave on every change: a crash never loses pairs
                saved = len(session.pairs)
            if act == "quit":
                break
    finally:
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m livefire.app measure", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--acquire-only", action="store_true", help="acquire, save the bundle and exit")
    ap.add_argument("--from-csv", default=None, metavar="pixels.csv",
                    help="no window: pairs from a CSV cam_a,u_a,v_a,cam_b,u_b,v_b,tape_cm")
    ap.add_argument("--pairs-out", default=None, help="pairs.csv path (default: next to the bundle / in --out)")
    ap.add_argument("--max-width", type=int, default=None)
    ap.add_argument("--max-height", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        bundle = resolve_bundle(a)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - acquisition problems end with a clear message, not a traceback
        print(f"acquisition failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if a.acquire_only:
        return 0
    pairs_path = Path(a.pairs_out) if a.pairs_out else Path(a.out) / time.strftime("measure_%Y%m%d_%H%M%S") / "pairs.csv"
    session = MeasureSession(bundle)
    if a.from_csv:
        try:
            n = session.load_pixel_pairs(a.from_csv)
        except (ValueError, KeyError, OSError) as e:
            print(f"measure failed: {e}", file=sys.stderr)
            return 2
        print(f"{n} pairs reconstructed from {a.from_csv}")
    else:
        try:
            run_window(session, pairs_path, a.max_width, a.max_height)
        except cv2.error as e:
            print(f"cannot open a window ({e}); use --from-csv for headless use", file=sys.stderr)
            return 1
    if not session.pairs:
        print("no pairs stored")
        return 0
    session.save(pairs_path)
    r = session.report()
    print(f"M4: n={int(r['n'])}  MAE {r['mae_cm']:.2f} cm  median {r['median_abs_cm']:.2f}  p95 {r['p95_abs_cm']:.2f}  "
          f"bias {r['bias_cm']:+.2f} cm")
    print(f"pairs saved to {pairs_path}; next: uv run python eval/geometry_mae.py --pairs {pairs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
