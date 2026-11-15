"""P3-002: segmentation-assisted acquisition on the synthetic table_v1 (same fixture style and noise as
tests/test_acquire.py: Oracle depth, 1 % noise, 3 % per-view scale error; # MOCK: oracle segmentation with permuted
per-camera instance ids)."""

import hashlib

import numpy as np
import pytest

from livefire.common.errors import SegmentationError
from livefire.perception.segment import OracleSegmentationBackend
from livefire.scene.acquire import acquire_with_report, keyframe_indices
from test_acquire import CAMS, GRID, SCENE, _backends, _cfg, _sources, _truth_dist

# Golden digest of the P1 (segmenters=None) two-camera acquisition of tests/test_acquire.py's fixture:
# sha256(obj_id.tobytes() + pos.tobytes() + material_id.tobytes()) over 8369 surfels, taken at commit 78a0ba5
# (P3-001), BEFORE acquire.py was edited for P3-002. Deterministic over repeated runs (checked 3x).
GOLDEN_P1_DIGEST = "f0597134a44008aa1eee903fd6907f43550a0a5b2a4001e8a3ecd0ccb8faafcf"
GOLDEN_P1_N = 8369


def _digest(snap):
    return hashlib.sha256(snap.obj_id.tobytes() + snap.pos.tobytes() + snap.material_id.tobytes()).hexdigest()


SURFACE_TOL_M = 0.007  # surface-position noise bound: 1 cm TSDF voxels + 1 % depth noise blur a 3 mm step (measured below)


def _membership(snap):
    """``{snapshot obj_id: (matched truth id, membership rate)}`` for every non-table object. Truth of a surfel =
    the nearest scene object, except that a non-table object within ``SURFACE_TOL_M`` wins over the table: the 3 mm
    cloth patch is thinner than the position noise, so a strict nearest-surface rule (tests/test_acquire.py's) calls
    about a third of the correctly labelled cloth surfels "table" (measured 0.63, see the report)."""
    d = _truth_dist(snap.pos)
    near = np.where(d[:, 1:].min(1) <= SURFACE_TOL_M, 1 + d[:, 1:].argmin(1), d.argmin(1))
    out = {}
    for o in sorted(set(snap.obj_id.tolist()) - {0}):
        m = snap.obj_id == o
        rates = np.bincount(near[m], minlength=5) / m.sum()
        tid = int(np.argmax(np.where(np.arange(5) == 0, 0, rates)))
        out[o] = (tid, float(rates[tid]))
    return out, near


def _oracles(cams, **kw):
    return {c: OracleSegmentationBackend(CAMS[c], SCENE, **kw) for c in cams}


def _run(cams, segmenters, **cfg_kw):
    srcs = _sources(cams)
    try:
        return acquire_with_report(srcs, {c: CAMS[c] for c in cams}, _backends(cams), _cfg(**cfg_kw), GRID,
                                   segmenters=segmenters)  # fmt: skip
    finally:
        for s in srcs:
            s.stop()


@pytest.fixture(scope="module")
def seg_two_cam():
    be = _oracles([0, 1])
    return _run([0, 1], be), be


def test_segmenters_none_is_bit_identical_to_p1():
    r = _run([0, 1], None)
    assert len(r.snapshot.pos) == GOLDEN_P1_N
    assert _digest(r.snapshot) == GOLDEN_P1_DIGEST
    assert "segment" not in r.timings_ms and "associate" not in r.timings_ms
    assert "segmentation" not in r.report


def test_cloth_separated_from_table_with_segmentation(seg_two_cam):
    """The geometric path merges the 3 mm cloth into the table (strict xfail in test_acquire.py); segmentation fixes it."""
    r, _ = seg_two_cam
    found, _near = _membership(r.snapshot)
    assert sorted(t for t, _ in found.values()) == [1, 2, 3, 4], found
    for o, (_tid, rate) in found.items():
        assert rate > 0.9, (o, found)
    assert len(r.snapshot.objects) == 5  # table + 4 objects
    snap = r.snapshot
    cloth = next(o for o, (t, _) in found.items() if t == 4)
    cl = SCENE.objects[4]  # footprint recall, independent of position noise: surfels on the patch are labelled cloth
    inside = ((snap.pos[:, 0] > cl.lo[0] + 0.01) & (snap.pos[:, 0] < cl.hi[0] - 0.01) & (snap.pos[:, 1] > cl.lo[1] + 0.01)
              & (snap.pos[:, 1] < cl.hi[1] - 0.01) & (snap.pos[:, 2] < 0.015) & (snap.normal[:, 2] > 0.9))  # fmt: skip
    assert inside.sum() > 100 and (snap.obj_id[inside] == cloth).mean() > 0.9
    far = (np.abs(snap.pos[:, 0]) < 0.7) & (np.abs(snap.pos[:, 1]) < 0.4) & (snap.normal[:, 2] > 0.9) & ~inside
    assert (snap.obj_id[far & (snap.pos[:, 2] < 0.01)] == 0).mean() > 0.9  # the rest of the tabletop stays table


def test_observations_are_the_permuted_per_camera_ids(seg_two_cam):
    r, be = seg_two_cam
    snap = r.snapshot
    found, _ = _membership(snap)
    table = snap.objects[0]
    assert table.name == "table" and table.material.category_hint == "table"
    for o in snap.objects:
        if o.obj_id == 0:
            truth = 0
        else:
            truth = found[o.obj_id][0]
        for c, inst in o.observations.items():
            assert inst == be[c].permutation[truth], (o.obj_id, truth, c, inst)
        assert set(o.observations) == {0, 1}, (o.obj_id, o.observations)
    assert len({tuple(sorted(be[c].permutation.items())) for c in be}) == 2  # ids really differ between cameras


def test_names_categories_and_report(seg_two_cam):
    r, _ = seg_two_cam
    snap, rep = r.snapshot, r.report
    cats = {o.obj_id: o.material.category_hint for o in snap.objects}
    assert sorted(c for c in cats.values()) == ["cloth", "metal cup", "paper sheet", "table", "wooden block"]
    for o in snap.objects:
        assert o.name == "table" if o.obj_id == 0 else o.name == f"{o.material.category_hint}_{o.obj_id}"
        assert o.material.source == "category_prior"
        if o.obj_id:  # materials are Task 3's: the P1 default still applies
            assert o.material.material_id == 0 and o.material.requires_review
    for o, d in rep["objects"].items():
        assert d["source"] == "segmentation" and d["categories"] == [cats[o]]
        assert set(d["observations"]) == {0, 1} and 0.0 <= d["unlabelled_fraction"] < 0.5
    assert rep["segmentation"]["n_instances"] >= 8 and rep["segmentation"]["links"]


def test_timings_include_segment_and_associate(seg_two_cam):
    t = seg_two_cam[0].timings_ms
    for k in ("capture", "depth", "accumulate", "tsdf", "extract", "objects", "segment", "associate", "snapshot", "total"):
        assert t[k] > 0, k


def test_single_camera_with_segmentation():
    r = _run([0], _oracles([0]))
    snap = r.snapshot
    found, _ = _membership(snap)
    assert sorted(t for t, _ in found.values()) == [1, 2, 3, 4], found
    assert all(rate > 0.9 for _t, rate in found.values()), found
    assert set(snap.view_depth) == {0}
    be = OracleSegmentationBackend(CAMS[0], SCENE)
    for o in snap.objects:
        assert set(o.observations) == {0}
        truth = 0 if o.obj_id == 0 else found[o.obj_id][0]
        assert o.observations[0] == be.permutation[truth]


def test_shared_backend_failure_falls_back_to_geometry():
    class _Broken:
        device, latency_ms, peak_vram_allocated_mb, peak_vram_reserved_mb = "cpu", 0.0, 0.0, 0.0

        def warmup(self, shape):
            pass

        def segment(self, frame):
            raise SegmentationError("model failed")

    r = _run([0, 1], _Broken())
    assert _digest(r.snapshot) == GOLDEN_P1_DIGEST
    assert any("segmentation failed" in w for w in r.report["warnings"])


def test_keyframe_indices():
    assert keyframe_indices(7, 1) == [3] and keyframe_indices(6, 1) == [2]
    assert keyframe_indices(6, 3) == [2, 3, 1] and keyframe_indices(2, 5) == [0, 1]




def test_two_keyframes_are_associated_like_extra_views():
    """``segment.keyframes = 2``: per-keyframe instance ids (view = cam + 1000 k) merge into the same 5 objects; the
    reported observation is the nearest-to-middle keyframe's id."""
    be = _oracles([0])
    r = _run([0], be, segment={"majority_window_px": 3, "keyframes": 2})
    snap = r.snapshot
    assert sorted(r.report["segmentation"]["views"]) == [0, 1000] and len(snap.objects) == 5
    found, _ = _membership(snap)
    assert sorted(t for t, _ in found.values()) == [1, 2, 3, 4]
    for o in snap.objects:
        truth = 0 if o.obj_id == 0 else found[o.obj_id][0]
        assert o.observations == {0: be[0].permutation[truth]}
