"""P1-005: accumulation, TSDF fusion, surface extraction, plane fit and clustering on small synthetic inputs."""

import numpy as np

from livefire.common.geometry import AABB, CameraCalibration, look_at, make_K
from livefire.fusion.accumulate import accumulate_depth
from livefire.fusion.extract import extract_surface
from livefire.fusion.objects import DROPPED, cluster_objects, fit_table_plane, load_acquisition_config
from livefire.fusion.tsdf import TSDFGrid, TSDFVolume


def test_accumulate_median_and_confidence():
    rng = np.random.default_rng(0)
    t, h, w = 9, 6, 7
    d = 2.0 + 0.002 * rng.normal(size=(t, h, w))
    v = np.ones((t, h, w), bool)
    d[:2, 0, 0] = 5.0  # 2 outlier frames
    v[:, 1, 1] = False  # never valid
    v[:7, 2, 2] = False  # valid in 2 of 9 frames only
    a = accumulate_depth(d.astype(np.float32), v, agree_tol_rel=0.02, min_confidence=0.5)
    assert abs(a.depth[0, 0] - 2.0) < 0.005 and abs(a.conf[0, 0] - 7 / 9) < 1e-6
    assert not a.valid[1, 1] and a.depth[1, 1] == 0.0 and a.conf[1, 1] == 0.0
    assert not a.valid[2, 2] and a.depth[2, 2] == 0.0
    assert a.valid[3, 3] and a.n_frames == t


def test_grid_centres_follow_contract():
    g = TSDFGrid(AABB(np.array([-0.8, -0.6, -0.05]), np.array([0.8, 0.6, 0.6])), 0.01)
    assert g.shape == (160, 120, 65)
    np.testing.assert_allclose(g.centers(np.array([[0, 0, 0]]))[0], [-0.795, -0.595, -0.045])


def _cam(pos):
    return CameraCalibration(0, 160, 120, make_K(150, 150, 79.5, 59.5), np.zeros(5), look_at(pos, (0, 0, 0)), 1)


def test_tsdf_plane_and_extraction():
    """A fronto-parallel-ish plane z = 0 seen by two cameras is recovered to < 2 mm with +z normals."""
    roi = AABB(np.array([-0.2, -0.2, -0.1]), np.array([0.2, 0.2, 0.2]))
    vol = TSDFVolume(roi, 0.01, 3.0, "cpu")
    for pos in ((0.3, -0.4, 0.8), (-0.4, 0.2, 0.7)):
        c = _cam(pos)
        from livefire.common.geometry import ray_dirs_camera

        d = ray_dirs_camera(c) @ c.T_world_from_camera[:3, :3].T
        t = -c.center_world[2] / d[..., 2]  # Z-depth == ray parameter
        vol.integrate(c, t.astype(np.float32), np.ones_like(t, dtype=np.float32))
    sp = extract_surface(vol.tsdf(), vol.weight(), vol.grid)
    inner = (np.abs(sp.pos[:, 0]) < 0.12) & (np.abs(sp.pos[:, 1]) < 0.12)
    assert inner.sum() > 200
    assert np.abs(sp.pos[inner, 2]).max() < 2e-3
    assert sp.normal[inner, 2].min() > 0.99
    assert abs(sp.footprint[inner].sum() - 0.24**2) / 0.24**2 < 0.15  # footprints add up to the area


def test_plane_fit_and_clustering():
    cfg = load_acquisition_config()
    rng = np.random.default_rng(1)
    g = np.stack(np.meshgrid(np.arange(-0.5, 0.5, 0.01), np.arange(-0.4, 0.4, 0.01)), -1).reshape(-1, 2)
    table = np.c_[g, 0.001 * rng.normal(size=len(g))]
    box = np.c_[rng.uniform(0.1, 0.2, 800), rng.uniform(0.1, 0.2, 800), rng.uniform(0.02, 0.08, 800)]
    speck = np.array([[-0.4, -0.3, 0.05], [-0.4, -0.3, 0.055]])
    skirt = np.c_[np.full(50, 0.5), np.linspace(-0.4, 0.4, 50), np.full(50, -0.03)]
    pos = np.concatenate([table, box, speck, skirt])
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    nrm[len(table) : len(table) + len(box)] = [1.0, 0.0, 0.0]
    nrm[-50:] = [1.0, 0.0, 0.0]
    fp = np.full(len(pos), 1e-4)
    pl = fit_table_plane(pos, nrm, fp, cfg)
    assert pl.agrees and pl.z_dev_m < 1e-3
    lab = cluster_objects(pos, nrm, fp, pl, cfg, 0.015)
    assert (lab[: len(table)] == 0).all() and (lab[-50:] == 0).all()
    assert (lab[len(table) : len(table) + 800] == 1).all()
    assert (lab[-52:-50] == DROPPED).all()


def test_plane_disagreement_warns(caplog):
    cfg = load_acquisition_config()
    g = np.stack(np.meshgrid(np.arange(-0.3, 0.3, 0.01), np.arange(-0.3, 0.3, 0.01)), -1).reshape(-1, 2)
    pos = np.c_[g, np.full(len(g), 0.03)]  # plane 3 cm above z = 0
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    pl = fit_table_plane(pos, nrm, np.full(len(pos), 1e-4), cfg)
    assert not pl.agrees
