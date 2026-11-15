"""P3-002: cross-view association on hand-built two-camera surfel sets (no renderer, no weights), plus the projection
visibility rule of ``label_surfels`` on a real (synthetic-scene) camera."""

import numpy as np
import pytest

from livefire.common.geometry import backproject
from livefire.fusion.accumulate import AccumulatedDepth
from livefire.fusion.associate import associate_objects, instance_colors, label_surfels
from livefire.fusion.objects import load_acquisition_config
from livefire.perception.types import SegmentationResult
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

CFG = load_acquisition_config()


def _labels(n, groups):
    """int32 (n,) with ``groups = {instance id: slice}``; -1 elsewhere."""
    a = np.full(n, -1, np.int32)
    for i, sl in groups.items():
        a[sl] = i
    return a


def _partition(labels, sl):
    return set(np.unique(labels[sl]).tolist())


def test_permuted_ids_merge_and_observations():
    n = 300
    A, B, C = slice(0, 100), slice(100, 200), slice(200, 300)
    cam0 = _labels(n, {0: A, 1: B, 2: C})
    cam1 = _labels(n, {2: A, 0: B, 1: C})  # same objects, different ids
    geo = np.zeros(n, np.int32)  # geometry knows nothing (everything "table")
    cats = {0: {0: "cup", 1: "box", 2: "table"}, 1: {2: "cup", 0: "box", 1: "table"}}
    r = associate_objects({0: cam0, 1: cam1}, geo, {}, CFG, categories=cats)
    assert len(set(r.labels.tolist())) == 3 and r.instance_labelled.all()
    for sl in (A, B, C):
        assert len(_partition(r.labels, sl)) == 1
    oa, ob, oc = (int(r.labels[s.start]) for s in (A, B, C))
    assert oc == 0  # the "table"-category object is obj 0, whatever the order of the ids
    assert r.observations[oa] == {0: 0, 1: 2} and r.observations[ob] == {0: 1, 1: 0} and r.observations[oc] == {0: 2, 1: 1}
    assert r.categories[oa] == ["cup"] and r.categories[ob] == ["box"] and r.categories[oc] == ["table"]
    assert {oa, ob} == {1, 2}


def test_instance_seen_in_one_camera_is_its_own_object():
    n = 200
    cam0 = _labels(n, {0: slice(0, 100), 5: slice(100, 200)})
    cam1 = _labels(n, {7: slice(0, 100)})  # the second object is invisible to camera 1
    r = associate_objects({0: cam0, 1: cam1}, np.zeros(n, np.int32), {}, CFG)
    both, only0 = int(r.labels[0]), int(r.labels[150])
    assert both != only0 and r.observations[both] == {0: 0, 1: 7} and r.observations[only0] == {0: 5}


def test_higher_iou_wins_and_loser_stays_separate():
    n = 100
    cam0 = _labels(n, {0: slice(0, 100)})
    cam1 = _labels(n, {0: slice(0, 60), 1: slice(60, 100)})  # IoU 0.6 and 0.4, both above min_iou
    r = associate_objects({0: cam0, 1: cam1}, np.zeros(n, np.int32), {}, CFG)
    assert r.observations[int(r.labels[0])] == {0: 0, 1: 0}
    assert r.observations[int(r.labels[99])] == {1: 1}  # the lower-IoU competitor is not pulled in
    # the surfels of the one-sided instance vote for their own object, not the shared one
    assert r.labels[0] != r.labels[99]


def test_colour_breaks_an_iou_tie():
    n = 100
    cam0 = _labels(n, {0: slice(0, 100)})
    cam1 = _labels(n, {0: slice(0, 52), 1: slice(52, 100)})  # IoU 0.52 vs 0.48: within iou_tie = 0.05
    red, green = np.array([50.0, 60.0, 40.0]), np.array([50.0, -60.0, 40.0])
    for colours, winner in (
        ({0: {0: green}, 1: {0: red, 1: green}}, 1),
        ({0: {0: red}, 1: {0: red, 1: green}}, 0),
    ):
        r = associate_objects({0: cam0, 1: cam1}, np.zeros(n, np.int32), colours, CFG)
        merged = r.observations[int(r.labels[0 if winner == 0 else 99])]
        assert merged == {0: 0, 1: winner}, (winner, r.observations)
    # outside the tie the colour is ignored: IoU 0.7 vs 0.3 picks instance 0 even though camera 1's colour matches 1
    cam1b = _labels(n, {0: slice(0, 70), 1: slice(70, 100)})
    r = associate_objects({0: cam0, 1: cam1b}, np.zeros(n, np.int32), {0: {0: green}, 1: {0: red, 1: green}}, CFG)
    assert r.observations[int(r.labels[0])] == {0: 0, 1: 0}


def test_unlabelled_surfels_keep_geometry_and_table_is_overridden_by_instances():
    rng = np.random.default_rng(0)
    n = 400
    pos = rng.random((n, 3))
    geo = np.zeros(n, np.int32)  # 0..199 table (incl. a flat "cloth" 100..149), 200..299 geometric obj 1, 300..399 obj 2
    geo[200:300], geo[300:400] = 1, 2
    cam0 = _labels(n, {3: slice(0, 100), 4: slice(100, 150), 8: slice(200, 250)})  # table, cloth, half of obj 1
    cam1 = _labels(n, {1: slice(0, 100), 2: slice(100, 150), 6: slice(200, 250)})
    cats = {0: {3: "table", 4: "cloth", 8: "box"}, 1: {1: "table", 2: "cloth", 6: "box"}}
    r = associate_objects({0: cam0, 1: cam1}, geo, {}, CFG, categories=cats, pos=pos)
    assert (r.labels[:100] == 0).all()
    assert len(_partition(r.labels, slice(100, 150))) == 1 and r.labels[100] != 0  # the cloth overrides the table label
    assert (r.labels[150:200] == 0).all() and not r.instance_labelled[150:200].any()  # unlabelled table stays table
    box = int(r.labels[200])
    assert (r.labels[200:300] == box).all() and r.instance_labelled[200:250].all() and not r.instance_labelled[250:300].any()
    other = int(r.labels[300])
    assert other not in (0, box, int(r.labels[100])) and (r.labels[300:] == other).all()  # no instance: own geometric object
    assert r.report["objects"][other]["source"] == "geometric" and r.report["objects"][other]["unlabelled_fraction"] == 1.0
    assert r.report["objects"][box]["source"] == "segmentation" and r.report["objects"][box]["unlabelled_fraction"] == 0.5
    assert r.observations[other] == {}


def test_objects_numbered_by_area_and_tiny_instances_ignored():
    n = 160
    cam0 = _labels(n, {0: slice(0, 100), 1: slice(100, 105), 2: slice(105, 160)})  # 100, 5 (<10 surfels), 55
    cam1 = _labels(n, {0: slice(0, 100), 1: slice(100, 105), 2: slice(105, 160)})
    geo = np.zeros(n, np.int32)
    area = np.ones(n)
    area[105:160] = 5.0  # the 55-surfel object is the larger one by area
    cats = {v: {0: "table", 1: "bit", 2: "box"} for v in (0, 1)}
    r = associate_objects({0: cam0, 1: cam1}, geo, {}, CFG, categories=cats, area=area)
    assert (r.labels[:100] == 0).all() and (r.labels[105:] == 1).all()
    assert not r.instance_labelled[100:105].any() and (r.labels[100:105] == 0).all()  # ignored: geometric label (table)


def test_geometric_table_fallback_without_a_table_instance():
    n = 300
    geo = np.zeros(n, np.int32)
    cam0 = _labels(n, {0: slice(0, 200), 1: slice(200, 260)})  # big tabletop instance, small cloth-like instance
    cam1 = _labels(n, {5: slice(0, 200), 6: slice(200, 260)})
    geo[:] = 0  # the geometric clustering called everything table (cloth merged)
    r = associate_objects({0: cam0, 1: cam1}, geo, {}, CFG)
    assert (r.labels[:200] == 0).all() and (r.labels[200:260] == 1).all()  # the cloth is not mistaken for the table


@pytest.fixture(scope="module")
def view0():
    cams, scene = load_cameras(), build_table_v1()
    cal = cams[0]
    v = render_view(scene, cal)
    acc = AccumulatedDepth(v.depth.astype(np.float32), v.valid.copy(), np.ones(v.depth.shape, np.float32), 1)
    return cal, v, acc


def test_visibility_rule(view0):
    cal, v, acc = view0
    ids = np.full(v.depth.shape, -1, np.int32)
    ids[300:420, 500:700] = 3  # a rectangular "instance" (the seg result is arbitrary here)
    seg = SegmentationResult(ids, {3: "thing"}, {3: 1.0}, cal.calib_version)
    uv = np.array([[600.0, 360.0], [550.0, 320.0], [650.0, 400.0], [100.0, 100.0]])
    z = np.array([v.depth[360, 600], v.depth[320, 550], v.depth[400, 650], v.depth[100, 100]], np.float64)
    on_surface = backproject(cal, uv, z)
    behind = backproject(cal, uv[:3], z[:3] + 0.05)  # 5 cm behind the visible surface: not visible (tol 2 cm)
    in_front = backproject(cal, uv[:3], z[:3] - 0.05)  # free space in front of the surface: not visible
    outside = backproject(cal, np.array([[-50.0, 10.0]]), np.array([1.0]))  # outside the image
    acc2 = AccumulatedDepth(acc.depth.copy(), acc.valid.copy(), acc.conf, 1)
    acc2.valid[360, 600] = False
    pos = np.concatenate([on_surface, behind, in_front, outside])
    out = label_surfels(pos, {0: cal}, {0: acc}, {0: seg}, CFG)[0]
    assert out.dtype == np.int32 and out.shape == (len(pos),)
    assert (out[:3] == 3).all()  # visible, inside the instance
    assert out[3] == -1  # visible but background pixel
    assert (out[4:] == -1).all()  # behind / in front of / outside: never take an id
    out2 = label_surfels(on_surface[:1], {0: cal}, {0: acc2}, {0: seg}, CFG)[0]
    assert out2[0] == -1  # invalid accumulated depth


def test_instance_colors_lab_means():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:, :2] = (255, 0, 0)
    rgb[:, 2:] = (0, 255, 0)
    ids = np.full((4, 4), -1, np.int32)
    ids[:, :2], ids[:2, 2:] = 0, 1
    c = instance_colors(rgb, ids)
    assert set(c) == {0, 1} and c[0][1] > 50 and c[1][1] < -50  # red: a* > 0, green: a* < 0
    assert instance_colors(rgb, np.full((4, 4), -1, np.int32)) == {}
