"""Volume renderer and compositor tests (P0-009) on the synthetic table_v1 rig. CPU Warp."""

import numpy as np
import pytest

from livefire.capture.types import Frame
from livefire.common.errors import CalibVersionError
from livefire.common.geometry import project
from livefire.render.blackbody import blackbody_lut, blackbody_rgb
from livefire.render.compositor import ArrayFireState, Compositor, blob_state
from livefire.scene.builder import snapshot_from_synthetic
from livefire.scene.synthetic import build_table_v1, render_view

G = 48
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@pytest.fixture(scope="module")
def rig():
    snap = snapshot_from_synthetic()
    scene = build_table_v1()
    frames = {i: render_view(scene, c, seed=0) for i, c in snap.cameras.items()}
    return snap, scene, frames, Compositor(snap, volume_scale=1.0)


def _centroid(L, mask_center=None, radius=None):
    lum = L @ LUMA
    if mask_center is not None:
        v, u = np.mgrid[0 : lum.shape[0], 0 : lum.shape[1]]
        lum = np.where((u - mask_center[0]) ** 2 + (v - mask_center[1]) ** 2 <= radius**2, lum, 0.0)
    s = lum.sum()
    v, u = np.mgrid[0 : lum.shape[0], 0 : lum.shape[1]]
    return np.array([(lum * u).sum() / s, (lum * v).sum() / s]), s


def _visible(snap, pts, margin=0.03):
    ok = np.ones(len(pts), bool)
    for cid, c in snap.cameras.items():
        uv, z, v = project(c, pts)
        u, w = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        inside = v & (u >= 60) & (u < c.width - 60) & (w >= 60) & (w < c.height - 60)
        d = snap.view_depth[cid][np.clip(w, 0, c.height - 1), np.clip(u, 0, c.width - 1)]
        ok &= inside & (d > z + margin) & ~snap.view_hole_mask[cid][np.clip(w, 0, c.height - 1), np.clip(u, 0, c.width - 1)]
    return ok


def test_cross_view_centroids(rig):
    snap, _, _, comp = rig
    rng = np.random.default_rng(1234)
    cand = np.column_stack([rng.uniform(-0.5, 0.5, 400), rng.uniform(-0.3, 0.3, 400), rng.uniform(0.3, 0.5, 400)])
    pts = cand[_visible(snap, cand)][:5]
    assert len(pts) == 5
    errs, rows = [], []
    for pt in pts:
        st = blob_state(snap, G, pt[None])
        row = []
        for cid, c in snap.cameras.items():
            _, L = comp.render_volume(cid, st)
            cen, _ = _centroid(L)
            e = float(np.linalg.norm(cen - project(c, pt)[0][0]))
            errs.append(e)
            row.append(e)
        rows.append(row)
    print("\ncentroid errors px (cam0, cam1) per point:", np.round(rows, 2).tolist())
    print(f"median {np.median(errs):.2f} px, max {np.max(errs):.2f} px; render ms/view {comp.last_volume_ms:.0f}")
    assert np.median(errs) < 6.0


def test_occlusion_behind_block(rig):
    snap, scene, _, comp = rig
    c0, c1 = snap.cameras[0], snap.cameras[1]
    block = scene.objects[1]
    centre = 0.5 * (np.array(block.lo) + np.array(block.hi))
    d = centre - c0.center_world
    d /= np.linalg.norm(d)
    lo, hi = np.array(block.lo), np.array(block.hi)
    a, b = (lo - c0.center_world) / d, (hi - c0.center_world) / d
    t_exit = np.maximum(a, b).min()  # ray exits the block's far face
    pt = c0.center_world + (t_exit + 0.1) * d
    print("\nocclusion point per brief (0.1 m past far face along the ray):", np.round(pt, 3))
    # The prescribed point lies below the table top (z<0): the centre ray leaves the 8 cm block at
    # z~0.008 heading down. Keep the same horizontal position, rest the blob on the table plane.
    assert pt[2] < 0
    pt[2] = 0.0
    st = blob_state(snap, G, pt[None], sigma_cells=1.0)
    _, L0 = comp.render_volume(0, st)
    _, L1 = comp.render_volume(1, st)
    r = 15
    (_, s0) = _centroid(L0, project(c0, pt)[0][0], r)
    (_, s1) = _centroid(L1, project(c1, pt)[0][0], r)
    print(f"occlusion ratio cam0/cam1 = {s0 / s1:.4f} (cam0 {s0:.3f}, cam1 {s1:.3f}); point {np.round(pt, 3)}")
    assert s1 > 0 and s0 < 0.05 * s1
    # control: with the occlusion clip disabled the same blob is clearly visible from cam0
    comp_nc = Compositor(snap, volume_scale=1.0)
    v0 = comp_nc.views[0]
    v0.t_far = np.where(v0.t_far > 0, 1e3, 0.0).astype(np.float32)  # ROI/active clip still applies in-kernel
    _, L0n = comp_nc.render_volume(0, st)
    (_, s0n) = _centroid(L0n, project(c0, pt)[0][0], r)
    print(f"control (occlusion clip off): cam0/cam1 = {s0n / s1:.2f}")
    assert s0n / s1 > 5


def test_holes_suppress_emission(rig):
    import dataclasses

    snap, _, _, comp = rig
    st = blob_state(snap, G, np.array([[0.0, 0.0, 0.3]]))
    _, L = comp.render_volume(0, st)
    assert L.sum() > 0
    holes = {0: np.ones_like(snap.view_hole_mask[0]), 1: snap.view_hole_mask[1]}
    snap_h = dataclasses.replace(snap, view_hole_mask=holes)
    T3, L3 = Compositor(snap_h, volume_scale=1.0).render_volume(0, st)
    assert L3.sum() == 0 and T3.min() == 1.0


def test_blackbody_lut():
    lut = blackbody_lut()
    assert lut.shape == (64, 3)
    assert np.allclose(lut @ LUMA, 1.0, atol=1e-4)
    r, g, b = blackbody_rgb(1000.0)
    assert r > g > b
    assert blackbody_rgb(3000.0)[2] / blackbody_rgb(3000.0)[0] > b / r


def test_round_trip_no_fire(rig):
    snap, _, frames, _ = rig
    comp = Compositor(snap)
    n = len(snap.pos)
    st = ArrayFireState(np.full((G, G, G), 293.15, np.float32), np.zeros((G, G, G), np.float32),
                        np.zeros(n, np.float32), np.zeros(n, np.float32), float(snap.roi.size[0] / G), snap.roi)  # fmt: skip
    for cid, c in snap.cameras.items():
        out = comp.render(cid, Frame(cid, frames[cid].rgb, 0, 0, c.calib_version), st)
        assert out.shape == frames[cid].rgb.shape and out.dtype == np.uint8 and out.flags.c_contiguous
        assert np.abs(out.astype(int) - frames[cid].rgb.astype(int)).max() <= 1


def test_char_and_errors(rig):
    snap, _, frames, comp = rig
    n = len(snap.pos)
    st = blob_state(snap, G, np.array([[5.0, 5.0, 5.0]]))  # off-ROI-centre, no active cells
    st.char[snap.obj_id == 1] = 1.0
    out = comp.render(0, Frame(0, frames[0].rgb, 0, 0, 1), st)
    m = frames[0].obj_id == 1
    assert out[m].astype(float).mean() < 0.6 * frames[0].rgb[m].astype(float).mean()
    with pytest.raises(KeyError):
        comp.render(7, Frame(7, frames[0].rgb, 0, 0, 1), st)
    with pytest.raises(CalibVersionError):
        comp.render(0, Frame(0, frames[0].rgb, 0, 0, 99), st)
    assert n == len(st.char)


def test_overlay_paper_solid(rig):
    """Paper elements at removed=1: no stipple from coplanar table surfels (owner-object rule)."""
    from scipy.ndimage import binary_erosion

    snap, _, frames, comp = rig
    removed = (snap.obj_id == 3).astype(np.float32)
    for cid in (0, 1):
        r = comp.views[cid].overlay.splat(removed)
        inner = binary_erosion(frames[cid].obj_id == 3, iterations=4)
        assert inner.sum() > 500
        frac = float((r[inner] > 0.9).mean())
        print(f"cam{cid}: paper interior px {inner.sum()}, removed>0.9: {frac:.4f}")
        assert frac >= 0.99
