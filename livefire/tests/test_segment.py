"""P3-001: segmentation backends. Oracle and helpers run without weights; real backends skip unless the extra and
cached weights exist (never downloads)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from livefire.capture.types import Frame
from livefire.common.errors import CalibVersionError, SegmentationError
from livefire.common.interfaces import SegmentationBackend
from livefire.perception.segment import (
    OracleSegmentationBackend,
    create_segmentation_backend,
    load_segment_config,
    result_from_masks,
    segment_extra_available,
    validate_segmentation_result,
)
from livefire.perception.segment.base import model_cache_dir
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

CAMS = load_cameras()
SCENE = build_table_v1()


def _frame(cam):
    v = render_view(SCENE, cam)
    return Frame(cam.cam_id, v.rgb, 0, 0, cam.calib_version), v


@pytest.mark.parametrize("cam_id", [0, 1])
def test_oracle_matches_ground_truth(cam_id):
    cam = CAMS[cam_id]
    fr, v = _frame(cam)
    be = OracleSegmentationBackend(cam, SCENE)
    assert isinstance(be, SegmentationBackend)
    be.warmup((cam.height, cam.width))
    res = be.segment(fr)
    validate_segmentation_result(res, (cam.height, cam.width), cam.calib_version)
    assert res.calib_version == cam.calib_version
    assert be.latency_ms > 0 and be.peak_vram_allocated_mb == 0.0 and be.peak_vram_reserved_mb == 0.0
    assert (res.instance_ids[v.obj_id < 0] == -1).all()  # room / miss stays background
    visible = {int(o) for o in np.unique(v.obj_id) if o >= 0}
    assert visible  # at least the table
    names = {i: n for i, (n, _m) in SCENE.objects_table().items()}
    cmap = load_segment_config()["category_map"]
    for oid in visible:
        inst = be.permutation[oid]
        assert inst in res.categories, f"object {oid} has no instance"
        gt = v.obj_id == oid
        pred = res.instance_ids == inst
        iou = (gt & pred).sum() / (gt | pred).sum()
        # A 1-px erosion removes a ring (IoU 0.94-0.99 for objects of 4000+ px, 0.37 for the 249 px edge-on box of
        # camera 0, measured); the brief's 0.95 holds for the exact erode_px=0 mask (tested below).
        assert iou >= (0.92 if gt.sum() >= 3000 else 0.3), (cam_id, names[oid], int(gt.sum()), iou)
        assert res.categories[inst] == cmap[names[oid]] and res.scores[inst] == 1.0


def test_oracle_ids_permuted_per_camera_and_consistent():
    perms = []
    for cam in CAMS.values():
        be = OracleSegmentationBackend(cam, SCENE)
        perm = be.permutation
        assert sorted(perm.values()) == sorted(perm)  # a permutation of the same id set
        assert perm != {k: k for k in perm}  # differs from world ids
        perms.append(perm)
        res = be.segment(_frame(cam)[0])
        assert set(res.categories) <= set(perm.values())
    assert perms[0] != perms[1]  # differs between cameras
    # seeded: same construction gives the same permutation
    assert OracleSegmentationBackend(CAMS[0], SCENE).permutation == perms[0]
    ident = OracleSegmentationBackend(CAMS[0], SCENE, permute_ids=False).permutation
    assert ident == {k: k for k in ident}


def test_oracle_erosion_and_drop_and_errors():
    cam = CAMS[0]
    fr, v = _frame(cam)
    raw = OracleSegmentationBackend(cam, SCENE, erode_px=0, permute_ids=False).segment(fr)
    np.testing.assert_array_equal(raw.instance_ids, v.obj_id)  # exact render (IoU 1.0 >= 0.95), room -1
    eroded = OracleSegmentationBackend(cam, SCENE, erode_px=2, permute_ids=False).segment(fr)
    assert (eroded.instance_ids >= 0).sum() < (raw.instance_ids >= 0).sum()
    none = OracleSegmentationBackend(cam, SCENE, drop_prob=1.0).segment(fr)
    assert (none.instance_ids == -1).all() and not none.categories
    with pytest.raises(CalibVersionError):
        OracleSegmentationBackend(cam, SCENE).segment(Frame(0, v.rgb, 0, 0, cam.calib_version + 1))
    with pytest.raises(SegmentationError):
        OracleSegmentationBackend(cam, SCENE).segment(Frame(0, v.rgb[:10], 0, 0, cam.calib_version))


def test_factory_selects_by_name_and_rejects_unknown():
    be = create_segmentation_backend("oracle", calib=CAMS[0], scene=SCENE)
    assert isinstance(be, OracleSegmentationBackend)
    with pytest.raises(SegmentationError, match="unknown segmentation backend"):
        create_segmentation_backend("nope")
    with pytest.raises(SegmentationError, match="calib"):
        create_segmentation_backend("oracle")
    cfg = load_segment_config()
    assert cfg["backend"] == "gdino_sam2" and cfg["max_instances"] == 32 and cfg["keyframes"] == 1
    assert cfg["box_threshold"]["value"] == 0.30 and "table" in cfg["vocabulary"]


def test_result_from_masks_resolves_overlap_by_score():
    a = np.zeros((6, 6), bool)
    b = np.zeros((6, 6), bool)
    a[:, :4] = True
    b[:, 2:] = True
    res = result_from_masks(np.stack([a, b]), [0.4, 0.9], ["low", "high"], 7)
    assert res.calib_version == 7 and res.instance_ids.dtype == np.int32
    assert res.categories == {0: "high", 1: "low"}  # renumbered by descending score
    assert res.scores == {0: pytest.approx(0.9), 1: pytest.approx(0.4)}
    assert (res.instance_ids[:, 2:] == 0).all()  # overlap goes to the higher score
    assert (res.instance_ids[:, :2] == 1).all()
    validate_segmentation_result(res, (6, 6), 7)


def test_result_from_masks_min_area_max_instances_and_empty():
    m = np.zeros((3, 8, 8), bool)
    m[0, :4, :4] = True  # 16 px
    m[1, 6:, 6:] = True  # 4 px
    m[2, 4:6, :3] = True  # 6 px
    res = result_from_masks(m, [0.9, 0.8, 0.7], ["a", "b", "c"], 1, min_area_px=5, max_instances=1)
    assert res.categories == {0: "a"} and (res.instance_ids >= 0).sum() == 16
    e = result_from_masks(np.zeros((0, 4, 4), bool), [], [], 3)
    assert (e.instance_ids == -1).all() and not e.categories
    with pytest.raises(SegmentationError):
        validate_segmentation_result(res, (4, 4), 1)


def test_core_imports_without_torch():
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None\n"
        "import livefire.perception.segment as s\n"
        "from livefire.perception.segment import factory, oracle, base\n"
        "assert s.segment_extra_available() is False\n"
        "from livefire.common.errors import SegmentationError\n"
        "try:\n    s.create_segmentation_backend('gdino_sam2')\n"
        "except SegmentationError as e:\n    assert 'perception' in str(e), e\n"
        "else:\n    raise SystemExit('no error')\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def _cached(hf_id: str) -> bool:
    snaps = model_cache_dir(load_segment_config()) / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any(any(p.iterdir()) for p in snaps.iterdir())  # a failed gated download leaves empty dirs


_REAL = [
    ("gdino_sam2", ["IDEA-Research/grounding-dino-tiny", "facebook/sam2.1-hiera-tiny"]),
    ("sam3", ["facebook/sam3"]),
    ("mask2former", ["facebook/mask2former-swin-tiny-coco-instance"]),
]


@pytest.mark.parametrize(("name", "ids"), _REAL)
def test_real_backend_output_contract(name, ids):
    if not segment_extra_available():
        pytest.skip("perception extra not installed")
    if not all(_cached(i) for i in ids):
        pytest.skip(f"weights for {name} not cached under data/models (run `make segment-spike`)")
    be = create_segmentation_backend(name, device="cpu")
    rgb = render_view(SCENE, CAMS[0]).rgb[::2, ::2].copy()  # one small image
    res = be.segment(Frame(0, rgb, 0, 0, 5))
    validate_segmentation_result(res, rgb.shape[:2], 5)
    be.release()
