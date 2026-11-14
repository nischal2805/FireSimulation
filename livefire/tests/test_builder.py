"""Snapshot builder tests (P0-006) on scene table_v1."""

import time

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from livefire.common.config import load_yaml
from livefire.common.geometry import project
from livefire.scene.builder import ViewInput, build_snapshot, manual_material
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

OBJ = range(5)


@pytest.fixture(scope="module")
def built():
    scene = build_table_v1()
    cams = load_cameras()
    rv = {i: render_view(scene, c, seed=0) for i, c in cams.items()}
    views = {i: ViewInput(cams[i], r.depth, r.valid, r.obj_id, r.rgb) for i, r in rv.items()}
    tab = scene.objects_table()
    mats = {o: manual_material(m) for o, (_, m) in tab.items()}
    t0 = time.perf_counter()
    snap = build_snapshot(views, mats, load_yaml("grid"), names={o: n for o, (n, _) in tab.items()})
    dt = time.perf_counter() - t0
    print(f"\nbuild time {dt:.2f} s, {len(snap.pos)} elements")
    return scene, cams, rv, snap


def test_elements_on_surface(built):
    scene, _, _, snap = built
    s = snap.surfel_spacing
    for o in OBJ:
        m = snap.obj_id == o
        assert m.sum() > 0, f"object {o} has no elements"
        d = scene.distance_to_object(o, snap.pos[m].astype(np.float64))
        assert d.max() <= s, (o, d.max())
        assert np.percentile(d, 95) <= 0.5 * s, (o, np.percentile(d, 95))


def _visible_area(scene, cams, rv, o, n=400_000):
    rng = np.random.default_rng(0)
    pts, _, total = scene.sample_surface(o, n, rng)
    vis = np.zeros(n, dtype=bool)
    for cid, c in cams.items():
        uv, z, ok = project(c, pts)
        u, v = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        ok &= (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)
        u, v = np.clip(u, 0, c.width - 1), np.clip(v, 0, c.height - 1)
        ok &= (rv[cid].obj_id[v, u] == o) & (np.abs(rv[cid].depth[v, u] - z) < 0.01)
        vis |= ok
    return total * vis.mean()


def test_area_within_20pct(built):
    scene, cams, rv, snap = built
    lines = []
    for o in OBJ:
        est = float(snap.area[snap.obj_id == o].sum())
        ref = _visible_area(scene, cams, rv, o)
        lines.append(f"obj {o}: area {est:.5f} analytic-visible {ref:.5f} err {100 * (est / ref - 1):+.1f}%")
        assert abs(est / ref - 1) < 0.20, lines[-1]
    print("\n" + "\n".join(lines))


def test_neighbors(built):
    snap = built[3]
    nb = snap.neighbors
    n = len(snap.pos)
    assert nb.shape[1] == load_yaml("grid")["neighbor_k_max"]
    ii = np.repeat(np.arange(n), nb.shape[1]).reshape(nb.shape)
    m = nb >= 0
    assert (nb[m] != ii[m]).all()
    assert (snap.obj_id[nb[m]] == snap.obj_id[ii[m]]).all()
    edges = set(zip(ii[m].tolist(), nb[m].tolist()))
    assert all((j, i) in edges for i, j in edges)
    assert m.sum() > 0
    assert (snap.neighbors >= 0).sum(1).mean() > 4


def test_materials(built):
    scene, _, _, snap = built
    tab = scene.objects_table()
    from livefire.perception.material.types import MATERIAL_CLASSES

    for o, (_, mat) in tab.items():
        m = snap.obj_id == o
        assert (snap.material_id[m] == MATERIAL_CLASSES.index(mat)).all()
        assert (snap.material_conf[m] == 1.0).all()
        assert snap.objects[o].material.source == "manual_override"


def test_view_depth_and_holes(built):
    _, _, rv, snap = built
    for cid, r in rv.items():
        obj = (r.obj_id >= 0) & r.valid
        # "no edge": all pixels in a 5x5 window have the same obj id and a small depth range
        inner = binary_erosion(obj, np.ones((5, 5), bool))
        d = r.depth
        same = np.ones_like(obj)
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                sh = np.roll(np.roll(r.obj_id, dy, 0), dx, 1)
                dd = np.roll(np.roll(d, dy, 0), dx, 1)
                same &= (sh == r.obj_id) & (np.abs(dd - d) < 0.02)
        m = inner & same & (snap.view_depth[cid] > 0)
        err = np.abs(snap.view_depth[cid][m] - d[m])
        med, p95 = np.median(err), np.percentile(err, 95)
        hole = snap.view_hole_mask[cid][obj].mean()
        print(f"\ncam{cid}: view_depth err median {med * 1e3:.2f} mm p95 {p95 * 1e3:.2f} mm "
              f"hole fraction (object pixels) {100 * hole:.3f}%")  # fmt: skip
        assert med < 0.01 and p95 < 0.03
        assert hole < 0.02
        room = r.valid & (r.obj_id < 0)
        assert not snap.view_hole_mask[cid][room].any()
        assert (snap.view_depth[cid][room] > 0).all()
        assert (snap.view_depth[cid][snap.view_hole_mask[cid]] == 0).all()


def test_obstacles(built):
    snap = built[3]
    G = snap.solid_mask.shape[0]
    h = snap.h_gas
    assert abs(h - 3.0 / G) < 1e-9
    assert snap.solid_mask.dtype == np.uint8 and snap.solid_fraction.dtype == np.float32
    for o, thick in [(0, True), (1, True), (2, True), (3, False), (4, False)]:
        m = snap.obj_id == o
        idx = np.floor((snap.pos[m] - snap.roi.min) / h).astype(int)
        sm = snap.solid_mask[idx[:, 0], idx[:, 1], idx[:, 2]]
        if thick:
            assert (sm == 1).all(), o
        else:
            # thin objects only produce solid cells where a thick object shares the cell
            thin_only = sm == 0
            assert thin_only.mean() > 0.5, o
    # paper cells away from the table are not solid
    m = snap.obj_id == 3
    hi = m & (snap.pos[:, 2] > 0.05)
    idx = np.floor((snap.pos[hi] - snap.roi.min) / h).astype(int)
    assert (snap.solid_mask[idx[:, 0], idx[:, 1], idx[:, 2]] == 0).all()
    sf = snap.solid_fraction
    assert sf.max() <= 1.0 and (sf[snap.solid_mask == 0] == 0).all() and (sf[snap.solid_mask == 1] > 0).all()
    # wood block surface cells solid (interior not required)
    w = snap.obj_id == 1
    idx = np.floor((snap.pos[w] - snap.roi.min) / h).astype(int)
    assert (snap.solid_mask[idx[:, 0], idx[:, 1], idx[:, 2]] == 1).all()


def test_snapshot_meta(built):
    snap = built[3]
    assert snap.calib_version == 1
    arrays = [snap.pos, snap.normal, snap.area, snap.obj_id, snap.material_id, snap.material_conf,
              snap.neighbors, snap.solid_mask, snap.solid_fraction,
              *snap.view_depth.values(), *snap.view_hole_mask.values()]  # fmt: skip
    assert all(not a.flags.writeable for a in arrays)
    assert np.allclose(np.linalg.norm(snap.normal, axis=1), 1.0, atol=1e-5)
    assert (snap.area > 0).all() and (snap.area <= 2 * snap.surfel_spacing**2 + 1e-9).all()
