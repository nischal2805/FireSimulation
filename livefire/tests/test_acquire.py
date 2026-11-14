"""P1-005: full acquisition path on the synthetic table_v1 (Oracle-style depth, per-view scale error, 1 % noise)."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from livefire.capture.file_source import VideoFileSource
from livefire.capture.synthetic_source import SyntheticCameraSource
from livefire.common.config import load_yaml
from livefire.fusion.objects import load_acquisition_config
from livefire.perception.depth import AnchoredDepthBackend, TableRegion
from livefire.perception.depth.base import RawDepth
from livefire.scene.acquire import acquire_with_report
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

SCALES = {0: 1.03, 1: 0.97}  # per-view scale error before anchoring
SCENE = build_table_v1()
CAMS = load_cameras()
TRUTH = {c: render_view(SCENE, CAMS[c]) for c in CAMS}
GRID = {**load_yaml("grid"), "G": 32}  # down-scaled gas grid keeps the test fast


class _NoisyOracle:
    """# MOCK: cached rendered ground-truth depth, scaled by ``scale`` with 1 % multiplicative noise per frame."""

    device = "cpu"
    peak_vram_allocated_mb = peak_vram_reserved_mb = 0.0

    def __init__(self, cam, scale, seed):
        self.t, self.scale, self.rng = TRUTH[cam], scale, np.random.default_rng(seed)

    def warmup(self, shape):
        pass

    def infer_raw(self, frame):
        z = self.scale * self.t.depth.astype(np.float64) * (1 + 0.01 * self.rng.normal(size=self.t.depth.shape))
        return RawDepth(np.where(self.t.valid, z, 0).astype(np.float32), self.t.valid.copy(), "metric", frame.calib_version)


def _backends(cams):
    reg = TableRegion.from_config()
    return {c: AnchoredDepthBackend(_NoisyOracle(c, SCALES[c], 7 + c), CAMS[c], reg) for c in cams}


def _cfg(**kw):
    cfg = load_acquisition_config()
    cfg.update(seconds=0.3, max_frames=6, **kw)
    return cfg


def _sources(cams):
    srcs = [SyntheticCameraSource(c, TRUTH[c].rgb, CAMS[c].calib_version, fps=200) for c in cams]
    for s in srcs:
        s.start()
    return srcs


@pytest.fixture(scope="module")
def two_cam():
    srcs = _sources([0, 1])
    try:
        return acquire_with_report(srcs, CAMS, _backends([0, 1]), _cfg(), GRID)
    finally:
        for s in srcs:
            s.stop()


def _truth_dist(pos):
    return np.stack([SCENE.distance_to_object(o.obj_id, pos.astype(float)) for o in SCENE.objects], 1)


def test_anchoring_and_geometry(two_cam):
    r, snap = two_cam.report, two_cam.snapshot
    for c in (0, 1):
        assert abs(r["cameras"][c]["anchor_scale_median"] - 1 / SCALES[c]) < 0.02
    assert r["plane"]["agrees"] and r["plane"]["z_dev_mm"] < 5.0
    top = (snap.obj_id == 0) & (snap.normal[:, 2] > 0.9) & (np.abs(snap.pos[:, 0]) < 0.7) & (np.abs(snap.pos[:, 1]) < 0.4)
    assert np.abs(np.median(snap.pos[top, 2])) < 5e-3
    d = _truth_dist(snap.pos).min(1)
    assert np.percentile(d, 95) < 0.01, np.percentile(d, 95)


def test_objects_and_membership(two_cam):
    snap = two_cam.snapshot
    d = _truth_dist(snap.pos)  # columns follow SCENE.objects order (obj_id 0..4)
    # membership: fraction of a cluster's elements whose nearest analytic surface belongs to the matched truth object
    near = d.argmin(1)
    truth_of = {}
    for o in sorted(set(snap.obj_id.tolist()) - {0}):
        m = snap.obj_id == o
        rates = np.bincount(near[m], minlength=5) / m.sum()
        tid = int(np.argmax(np.where(np.arange(5) == 0, 0, rates)))
        truth_of[o] = tid
        assert rates[tid] > 0.9, (o, rates)
    # wood (1), metal (2), paper (3) are found. The cloth patch (4) lies 3 mm above the table: below the table-plane
    # tolerance (and the depth noise), so it merges with the table (documented limit of geometric clustering, P1R4).
    assert sorted(truth_of.values()) == [1, 2, 3]
    assert len(snap.objects) == 4
    top = (snap.obj_id == 0) & (snap.normal[:, 2] > 0.9)
    assert np.isin(near[top], [0, 4]).mean() > 0.9  # cloth lies on the table


def test_snapshot_contract_and_materials(two_cam):
    snap = two_cam.snapshot
    assert snap.calib_version == CAMS[0].calib_version and set(snap.cameras) == {0, 1}
    assert snap.objects[0].name == "table" and snap.objects[0].material.material_id == 7
    assert not snap.objects[0].material.requires_review
    for o in snap.objects:
        if o.obj_id:
            assert o.material.material_id == 0 and o.material.requires_review
            assert set(o.observations) <= {0, 1} and o.observations
    nb = snap.neighbors
    ii = np.repeat(np.arange(len(snap.pos)), nb.shape[1]).reshape(nb.shape)
    m = nb >= 0
    assert (snap.obj_id[nb[m]] == snap.obj_id[ii[m]]).all()
    edges = set(zip(ii[m].tolist(), nb[m].tolist()))
    assert all((j, i) in edges for i, j in edges)
    assert snap.solid_mask.sum() > 0
    for c in (0, 1):
        vd, hole = snap.view_depth[c], snap.view_hole_mask[c]
        assert (vd[hole] == 0).all() and (vd[~hole] > 0).all()
        room = TRUTH[c].valid & (TRUTH[c].obj_id < 0)
        assert hole[room].mean() < 1e-4  # a few isolated low-confidence pixels (noise) may remain holes
        obj = TRUTH[c].obj_id == 1  # wood block: splat depth agrees with truth
        both = obj & (vd > 0)
        assert both.sum() > 100 and np.median(np.abs(vd[both] - TRUTH[c].depth[both])) < 0.01


def test_timings_reported(two_cam):
    t = two_cam.timings_ms
    for k in ("capture", "depth", "accumulate", "tsdf", "extract", "objects", "snapshot", "total"):
        assert t[k] >= 0
    assert two_cam.report["acquisition_s"] == pytest.approx(t["total"] / 1e3)


def test_single_camera():
    srcs = _sources([0])
    try:
        r = acquire_with_report(srcs, {0: CAMS[0]}, _backends([0]), _cfg(), GRID)
    finally:
        for s in srcs:
            s.stop()
    snap = r.snapshot
    assert set(snap.view_depth) == {0} and len(snap.objects) >= 2
    d = _truth_dist(snap.pos).min(1)
    assert np.percentile(d, 95) < 0.015


@pytest.mark.slow
def test_recorded_clip_acceptance(tmp_path):
    """Recorded synthetic clips (VideoFileSource) + default cfg (5 s window, 1 cm voxels): within seconds + 30 s (CPU)."""
    srcs = []
    for c in (0, 1):
        p = tmp_path / f"cam{c}.mp4"
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
        for _ in range(180):
            wr.write(cv2.cvtColor(TRUTH[c].rgb, cv2.COLOR_RGB2BGR))
        wr.release()
        srcs.append(VideoFileSource(c, p, realtime=True, loop=True, calib_version=CAMS[c].calib_version))
        srcs[-1].start()
    try:
        cfg = load_acquisition_config()
        r = acquire_with_report(srcs, CAMS, _backends([0, 1]), cfg, GRID)
    finally:
        for s in srcs:
            s.stop()
    assert r.report["acquisition_s"] < cfg["seconds"]["value"] + 30
    d = _truth_dist(r.snapshot.pos).min(1)
    r.report["geometry_p95_mm"] = float(1e3 * np.percentile(d, 95))
    out = os.environ.get("LIVEFIRE_WRITE_EVAL")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"label": "CPU, warp-cpu, oracle+noise MOCK", **r.report}, indent=1, default=str))
