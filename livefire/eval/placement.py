"""Camera placement study (P1-007, PRD 7.1): is the rig good enough to trust fusion?

From the stored calibration and a first acquisition (an ``AcqBundle``) it measures, on the table plane ``z = 0``
inside the table area under study (default: XY footprint of ``acquisition.fusion_roi``):

* **angle** between each camera pair, seen from the table-area centre ``(cx, cy, 0)`` (3-D angle between the two
  centre-to-camera vectors; 0 deg = same place, 180 deg = opposite sides);
* per camera: distance to the table centre, ``in_frame`` fraction, ``valid_coverage`` (in-frame table samples with valid,
  non-hole frozen depth), ``visible_fraction`` (in frame, valid depth and depth consistent with the plane, i.e. not
  hidden by an object);
* ``shared_fraction``: table samples visible in **every** camera.

It fails loudly (exit code 1, concrete guidance printed) when a threshold of ``configs/placement.yaml`` is violated.
One camera (P1R1) is allowed: the angle and shared checks are skipped with a warning. Output: CSV
``eval/results/placement_<label>.csv`` (one row per metric).

Usage: ``uv run python eval/placement.py [--bundle acquisition.npz | --synthetic] [--region x0,y0,x1,y1] [--out-dir eval/results]``
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from livefire.app.acq_bundle import AcqBundle, add_source_args, resolve_bundle
from livefire.common.config import Params, load_yaml
from livefire.common.eval_csv import write_csv
from livefire.common.geometry import project

FIELDS = ("metric", "scope", "value", "unit", "threshold", "status", "note")


class PlacementError(RuntimeError):
    """The camera placement is below the configured thresholds; the message carries concrete guidance."""


@dataclass
class PlacementReport:
    """``ok`` is False when any entry of ``failures`` exists. ``rows`` are CSV rows ``FIELDS``; ``angles_deg``
    ``{(cam_a, cam_b): deg}``; ``per_cam`` ``{cam: {distance_m, in_frame, valid_coverage, visible_fraction}}``;
    ``shared_fraction`` None with one camera. ``guidance`` lists one concrete action per failure."""

    ok: bool
    failures: list[str]
    warnings: list[str]
    guidance: list[str]
    angles_deg: dict[tuple[int, int], float]
    per_cam: dict[int, dict[str, float]]
    shared_fraction: float | None
    n_samples: int
    region_xy: tuple[float, float, float, float]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise PlacementError("camera placement is not good enough for fusion:\n  - " + "\n  - ".join(self.guidance))


def load_placement_params() -> Params:
    """``configs/placement.yaml: placement`` entries with a ``{value, unit, range, status}`` schema (the ``region_xy_m``
    key is returned separately by ``placement_study``)."""
    raw = dict(load_yaml("placement")["placement"])
    raw.pop("region_xy_m", None)
    return Params(raw)


def table_samples(region_xy: tuple[float, float, float, float], spacing: float) -> np.ndarray:
    """(N,3) world points on ``z = 0`` on a regular grid over ``region_xy = (x0, y0, x1, y1)`` m (cell centres)."""
    x0, y0, x1, y1 = region_xy
    nx, ny = max(1, round((x1 - x0) / spacing)), max(1, round((y1 - y0) / spacing))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)


def _visibility(bundle: AcqBundle, cam: int, pts: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(in_frame, valid, visible)`` bool (N,) for table samples ``pts`` in camera ``cam`` of the bundle."""
    cal = bundle.cameras[cam]
    uv, z, front = project(cal, pts)
    u, v = np.floor(uv[:, 0] + 0.5).astype(int), np.floor(uv[:, 1] + 0.5).astype(int)
    inside = front & (u >= 0) & (u < cal.width) & (v >= 0) & (v < cal.height)
    uc, vc = np.where(inside, u, 0), np.where(inside, v, 0)
    d = bundle.view_depth[cam][vc, uc].astype(np.float64)
    valid = inside & ~bundle.view_hole[cam][vc, uc] & (d > 0)
    visible = valid & (np.abs(d - z) <= tol)
    return inside, valid, visible


def placement_study(bundle: AcqBundle, *, region_xy: tuple[float, float, float, float] | None = None,
                    params: Params | None = None) -> PlacementReport:
    """Run the study on ``bundle`` (see module docstring). ``region_xy`` overrides the table area; thresholds from
    ``configs/placement.yaml`` unless ``params`` is given. Never raises for a bad placement: inspect ``ok`` or call
    ``raise_if_failed``."""
    cfg = load_yaml("placement")["placement"]
    p = params or load_placement_params()
    if region_xy is None:
        r = cfg.get("region_xy_m")
        region_xy = tuple(float(x) for x in r) if r else (float(bundle.roi.min[0]), float(bundle.roi.min[1]),
                                                           float(bundle.roi.max[0]), float(bundle.roi.max[1]))
    pts = table_samples(region_xy, p["sample_spacing_m"])
    centre = np.array([(region_xy[0] + region_xy[2]) / 2, (region_xy[1] + region_xy[3]) / 2, 0.0])
    cams = sorted(bundle.cameras)
    failures: list[str] = []
    warnings: list[str] = []
    guidance: list[str] = []
    rows: list[dict[str, Any]] = []

    def row(metric: str, scope: str, value: float, unit: str, thr: str = "", status: str = "info", note: str = "") -> None:
        rows.append({"metric": metric, "scope": scope, "value": value, "unit": unit, "threshold": thr, "status": status, "note": note})

    def fail(msg: str, advice: str) -> None:
        failures.append(msg)
        guidance.append(f"{msg}. {advice}")

    # ---- per camera
    per_cam: dict[int, dict[str, float]] = {}
    vis_masks: dict[int, np.ndarray] = {}
    for c in cams:
        inside, valid, visible = _visibility(bundle, c, pts, p["visible_depth_tol_m"])
        vis_masks[c] = visible
        dist = float(np.linalg.norm(bundle.cameras[c].center_world - centre))
        per_cam[c] = {"distance_m": dist, "in_frame": float(inside.mean()),
                      "valid_coverage": float(valid.sum() / inside.sum()) if inside.any() else 0.0,
                      "visible_fraction": float(visible.mean())}  # fmt: skip
        pc = per_cam[c]
        row("distance_to_table_centre", f"cam{c}", dist, "m", f"{p['distance_rec_min_m']}-{p['distance_rec_max_m']} (recommended)",
            "info")  # fmt: skip
        if not p["distance_rec_min_m"] <= dist <= p["distance_rec_max_m"]:
            warnings.append(f"cam{c}: {dist:.2f} m from the table centre (recommended {p['distance_rec_min_m']:.1f}-"
                            f"{p['distance_rec_max_m']:.1f} m)")  # fmt: skip
        row("in_frame_fraction", f"cam{c}", pc["in_frame"], "-")
        ok = pc["valid_coverage"] >= p["min_valid_coverage"]
        row("valid_coverage", f"cam{c}", pc["valid_coverage"], "-", f">= {p['min_valid_coverage']}", "pass" if ok else "FAIL",
            "in-frame table samples with valid frozen depth / in-frame table samples")  # fmt: skip
        if not ok:
            fail(f"cam{c}: only {pc['valid_coverage']:.0%} of the table it sees has valid depth (need {p['min_valid_coverage']:.0%})",
                 "improve lighting, remove reflective or textureless surfaces from view, re-run calibration, or move the camera closer")  # fmt: skip
        ok = pc["visible_fraction"] >= p["min_visible_fraction"]
        row("visible_fraction", f"cam{c}", pc["visible_fraction"], "-", f">= {p['min_visible_fraction']}", "pass" if ok else "FAIL",
            "table samples in frame, valid depth and not hidden / all table samples")  # fmt: skip
        if not ok:
            fail(f"cam{c}: sees only {pc['visible_fraction']:.0%} of the table area (need {p['min_visible_fraction']:.0%})",
                 "aim the camera at the table centre, lower the zoom or move it back, and keep objects from blocking the view")  # fmt: skip

    # ---- pairs
    angles: dict[tuple[int, int], float] = {}
    for a, b in itertools.combinations(cams, 2):
        va, vb = bundle.cameras[a].center_world - centre, bundle.cameras[b].center_world - centre
        ang = float(np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)), -1.0, 1.0))))
        ang = round(ang, 6)  # arccos round-off (60 deg -> 59.99999999999999) must not cross a band edge
        angles[(a, b)] = ang
        ok = p["angle_min_deg"] <= ang <= p["angle_max_deg"]
        row("camera_angle", f"cam{a}-cam{b}", ang, "deg", f"{p['angle_min_deg']}-{p['angle_max_deg']}", "pass" if ok else "FAIL",
            "3-D angle at the table-area centre between the two camera positions")  # fmt: skip
        if ang < p["angle_min_deg"]:
            fail(f"cam{a} and cam{b} are only {ang:.0f} deg apart (need {p['angle_min_deg']:.0f}-{p['angle_max_deg']:.0f})",
                 "move one camera sideways around the table so the two view directions differ by 60-90 degrees; "
                 "nearly parallel views give poor depth fusion and large occluded areas")  # fmt: skip
        elif ang > p["angle_max_deg"]:
            fail(f"cam{a} and cam{b} are {ang:.0f} deg apart (limit {p['angle_max_deg']:.0f})",
                 "bring the cameras closer together (60-90 degrees): near-opposite views see different sides of objects and "
                 "share little of the same surface")  # fmt: skip
        elif not p["angle_rec_min_deg"] <= ang <= p["angle_rec_max_deg"]:
            warnings.append(f"cam{a}-cam{b}: {ang:.0f} deg is allowed but outside the recommended "
                            f"{p['angle_rec_min_deg']:.0f}-{p['angle_rec_max_deg']:.0f} deg")  # fmt: skip

    # ---- shared
    shared: float | None = None
    if len(cams) >= 2:
        shared = float(np.logical_and.reduce([vis_masks[c] for c in cams]).mean())
        ok = shared >= p["min_shared_fraction"]
        row("shared_fraction", "all", shared, "-", f">= {p['min_shared_fraction']}", "pass" if ok else "FAIL",
            "table samples visible in every camera / all table samples")  # fmt: skip
        if not ok:
            fail(f"only {shared:.0%} of the table area is seen by all cameras (need {p['min_shared_fraction']:.0%})",
                 "aim both cameras at the same table centre, keep the objects of interest inside the overlap, "
                 "and check that nothing blocks either view")  # fmt: skip
    else:
        warnings.append("single camera: angle and shared-area checks skipped (single-view fallback, no occlusion cover)")
    row("table_samples", "all", float(len(pts)), "count", "", "info",
        f"region x0,y0,x1,y1 = {','.join(f'{x:.3f}' for x in region_xy)} m")  # fmt: skip
    return PlacementReport(not failures, failures, warnings, guidance, angles, per_cam, shared, len(pts), region_xy, rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--region", default=None, metavar="x0,y0,x1,y1",
                    help="table area in metres, world frame (default: XY footprint of acquisition.fusion_roi)")
    ap.add_argument("--label", default=None, help="output file name part (default: derived from the bundle label); "
                    "use one per scene so a second run does not overwrite the first")
    ap.add_argument("--out-dir", default="eval/results")
    a = ap.parse_args(argv)
    try:
        region = tuple(float(x) for x in a.region.split(",")) if a.region else None
        if region is not None and (len(region) != 4 or region[2] <= region[0] or region[3] <= region[1]):
            raise ValueError("needs x0,y0,x1,y1 with x1 > x0 and y1 > y0")
    except ValueError as e:
        print(f"bad --region {a.region!r}: {e} (write it as --region=\"-0.5,-0.4,0.5,0.4\")", file=sys.stderr)
        return 2
    try:
        bundle = resolve_bundle(a)
        rep = placement_study(bundle, region_xy=region)  # type: ignore[arg-type]
    except (ValueError, OSError, RuntimeError) as e:  # calibration, camera, depth or reconstruction problems
        print(f"placement study failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    label = a.label or _slug(bundle.label)
    path = write_csv(Path(a.out_dir) / f"placement_{label}.csv", FIELDS, rep.rows)
    print(f"placement study ({bundle.label}); region {rep.region_xy}, {rep.n_samples} samples")
    for r in rep.rows:
        print(f"  {r['metric']:26s} {r['scope']:10s} {r['value']:.3f} {r['unit']:5s} {r['status']}")
    for w in rep.warnings:
        print(f"WARNING: {w}")
    print(f"wrote {path}")
    if not rep.ok:
        print("PLACEMENT FAILED:", file=sys.stderr)
        for g in rep.guidance:
            print(f"  - {g}", file=sys.stderr)
        return 1
    print("placement OK")
    return 0


def _slug(s: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "run"


if __name__ == "__main__":
    sys.exit(main())
