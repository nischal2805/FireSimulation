"""Oracle segmentation backend: exact instance masks from the synthetic renderer (tests and sanity runs only)."""

from __future__ import annotations

import cv2
import numpy as np

from livefire.common.errors import CalibVersionError, SegmentationError
from livefire.common.geometry import CameraCalibration
from livefire.perception.segment.base import MetricsMixin, load_segment_config, result_from_masks
from livefire.perception.types import SegmentationResult
from livefire.scene.synthetic import SyntheticScene, build_table_v1, render_view


class OracleSegmentationBackend(MetricsMixin):
    """# MOCK: ground-truth instances rendered from a ``SyntheticScene`` through the camera ``calib``.

    The frame pixels are ignored (only ``calib_version`` and size are checked). ``render_view(...).obj_id`` is turned
    into one mask per real object (the room, obj_id -1, stays background -1). Seeded impairments
    (``numpy.random.default_rng([seed, calib.cam_id])``): each mask is eroded ``erode_px`` times with a 3x3 kernel,
    objects are dropped with probability ``drop_prob`` (missed detections), and with ``permute_ids`` the per-camera
    instance ids are a seeded permutation of the world obj_ids (``self.permutation``: world obj_id -> instance id),
    so they differ from world ids and between cameras (exercises the cross-view association). Categories come from
    ``category_map`` (scene object name -> vocabulary phrase, default ``configs/segment.yaml``), scores are 1.0.
    Output host NumPy, caller-owned; CPU only (``device = "cpu"``).
    """

    def __init__(
        self,
        calib: CameraCalibration,
        scene: SyntheticScene | None = None,
        *,
        erode_px: int | None = None,
        drop_prob: float | None = None,
        permute_ids: bool | None = None,
        seed: int | None = None,
        category_map: dict[str, str] | None = None,
        cfg: dict | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or load_segment_config()
        oc = cfg.get("oracle", {})
        self.calib = calib
        self.scene = scene or build_table_v1()
        self.erode_px = int(oc.get("erode_px", 1) if erode_px is None else erode_px)
        self.drop_prob = float(oc.get("drop_prob", 0.0) if drop_prob is None else drop_prob)
        self.permute_ids = bool(oc.get("permute_ids", True) if permute_ids is None else permute_ids)
        self.seed = int(oc.get("seed", 1234) if seed is None else seed)
        self.category_map = dict(cfg.get("category_map", {}) if category_map is None else category_map)
        self.device = "cpu"
        self._rng = np.random.default_rng([self.seed, int(calib.cam_id)])
        ids = sorted(self.scene.objects_table())
        if self.permute_ids:
            perm = np.random.default_rng([self.seed, int(calib.cam_id), 7]).permutation(len(ids))
        else:
            perm = np.arange(len(ids))
        self.permutation: dict[int, int] = {i: int(p) for i, p in zip(ids, perm, strict=True)}

    def warmup(self, shape: tuple[int, int]) -> None:
        """No-op (nothing to compile)."""

    def _category(self, name: str) -> str:
        return self.category_map.get(name, name.replace("_", " "))

    def segment(self, frame) -> SegmentationResult:
        """Instance map of the rendered scene (host NumPy; ids = ``permutation`` of the world obj_ids)."""
        if frame.calib_version != self.calib.calib_version:
            raise CalibVersionError(f"frame calib_version {frame.calib_version} != {self.calib.calib_version}")
        if frame.rgb.shape[:2] != (self.calib.height, self.calib.width):
            raise SegmentationError(f"frame {frame.rgb.shape[:2]} does not match calibration")

        def run() -> SegmentationResult:
            view = render_view(self.scene, self.calib)
            info = self.scene.objects_table()
            kernel = np.ones((3, 3), np.uint8)
            masks, cats, inv = [], [], {}
            for oid in sorted(info):
                m = view.obj_id == oid
                if not m.any():
                    continue
                if self.drop_prob > 0.0 and self._rng.random() < self.drop_prob:
                    continue
                if self.erode_px > 0:
                    m = cv2.erode(m.astype(np.uint8), kernel, iterations=self.erode_px).astype(bool)
                    if not m.any():
                        continue
                inv[len(masks)] = oid
                masks.append(m)
                cats.append(self._category(info[oid][0]))
            h, w = view.obj_id.shape
            stack = np.stack(masks) if masks else np.zeros((0, h, w), bool)
            res = result_from_masks(stack, np.ones(len(masks)), cats, frame.calib_version)
            # result_from_masks numbers instances by score (all 1.0 -> input order); remap to the permuted ids.
            ids = np.full_like(res.instance_ids, -1)
            categories: dict[int, str] = {}
            for k, c in res.categories.items():
                new = self.permutation[inv[k]]
                ids[res.instance_ids == k] = new
                categories[new] = c
            return SegmentationResult(ids, categories, {k: 1.0 for k in categories}, frame.calib_version)

        return self._timed(run)
