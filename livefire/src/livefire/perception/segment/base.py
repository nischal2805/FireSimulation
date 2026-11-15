"""Shared pieces of the segmentation backends: result construction, validation, config and availability helpers.

Frames: image space only (camera OpenCV pixels (u right, v down)); no world frame is involved. Results live in host
NumPy memory and are owned by the caller. Instance ids are per-camera observation ids (PRD 6.1), never world ids.
Imports without torch. ``MetricsMixin`` is reused from the depth package (same timer and VRAM semantics).
"""

from __future__ import annotations

import numpy as np

from livefire.common.config import load_yaml
from livefire.common.errors import SegmentationError
from livefire.perception.depth.base import MetricsMixin, default_device, model_cache_dir
from livefire.perception.types import SegmentationResult

__all__ = [
    "MetricsMixin",
    "dedup_masks",
    "default_device",
    "load_segment_config",
    "model_cache_dir",
    "result_from_masks",
    "scalar",
    "segment_extra_available",
    "validate_segmentation_result",
]


def segment_extra_available() -> bool:
    """True when torch, torchvision, transformers and pillow can be imported (the ``perception`` extra)."""
    try:
        import PIL  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def load_segment_config() -> dict:
    """``configs/segment.yaml`` as a dict."""
    return load_yaml("segment")


def scalar(entry) -> float:
    """Value of a ``{value, unit, range, status}`` parameter entry (or a bare number)."""
    return float(entry["value"] if isinstance(entry, dict) else entry)


def dedup_masks(masks: np.ndarray, scores, iou_thresh: float) -> np.ndarray:
    """Indices of the masks kept (in descending score order) after removing near-duplicates: walking from the best score down, a mask is dropped when its IoU with an already kept mask is
    ``>= iou_thresh``. ``masks`` bool (N, H, W) host, ``scores`` (N,). Pure; ties keep the lower input index."""
    masks = np.asarray(masks, dtype=bool)
    n = masks.shape[0]
    order = np.argsort(-np.asarray(scores, dtype=np.float64).reshape(n), kind="stable")
    kept: list[int] = []
    for i in order:
        dup = False
        for j in kept:
            union = int((masks[i] | masks[j]).sum())
            if union and int((masks[i] & masks[j]).sum()) / union >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def result_from_masks(
    masks: np.ndarray,
    scores,
    categories,
    calib_version: int,
    *,
    min_area_px: int = 0,
    max_instances: int | None = None,
) -> SegmentationResult:
    """Build a contract-clean ``SegmentationResult`` from per-instance boolean masks.

    ``masks`` bool (N, H, W) host (candidate instances, possibly overlapping); ``scores`` (N,) in [0, 1];
    ``categories`` N strings. Overlaps are resolved by score: each pixel goes to the highest-scoring mask covering it
    (ties: lower input index). Instances whose remaining area is below ``min_area_px`` are dropped, then at most
    ``max_instances`` of the highest scores are kept. Ids are renumbered ``0..n-1`` by descending score. Returns an
    all ``-1`` map for N = 0. Pure, caller owns the result.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"masks must be (N,H,W), got {masks.shape}")
    n, h, w = masks.shape
    scores = np.clip(np.asarray(scores, dtype=np.float64).reshape(n), 0.0, 1.0)
    cats = list(categories)
    if len(cats) != n:
        raise ValueError("categories must have one entry per mask")
    order = np.argsort(-scores, kind="stable")  # best first; stable keeps the lower index on ties
    owner = np.full((h, w), -1, dtype=np.int32)
    for rank in range(n - 1, -1, -1):  # paint worst first so the best score overwrites
        owner[masks[order[rank]]] = order[rank]
    keep: list[int] = []
    for i in order:
        if int((owner == i).sum()) >= max(int(min_area_px), 1):
            keep.append(int(i))
    if max_instances is not None:
        keep = keep[: int(max_instances)]
    ids = np.full((h, w), -1, dtype=np.int32)
    out_cat: dict[int, str] = {}
    out_score: dict[int, float] = {}
    for new_id, i in enumerate(keep):
        ids[owner == i] = new_id
        out_cat[new_id] = str(cats[i])
        out_score[new_id] = float(scores[i])
    return SegmentationResult(np.ascontiguousarray(ids), out_cat, out_score, int(calib_version))


def validate_segmentation_result(res: SegmentationResult, shape: tuple[int, int], calib_version: int) -> None:
    """Check the ``SegmentationBackend`` output contract; raises ``SegmentationError`` on the first violation.

    int32 C-contiguous (H, W) ``instance_ids`` with -1 background, every non-negative id present in ``categories``
    and ``scores`` (and vice versa, for ids that appear), scores in [0, 1], ``calib_version`` copied.
    """
    a = res.instance_ids
    if a.dtype != np.int32 or a.shape != tuple(shape) or not a.flags.c_contiguous:
        raise SegmentationError(f"instance_ids must be C-contiguous int32 {tuple(shape)}, got {a.dtype} {a.shape}")
    if int(a.min(initial=0)) < -1:
        raise SegmentationError("instance_ids below -1")
    present = {int(i) for i in np.unique(a) if i >= 0}
    if present - set(res.categories) or present - set(res.scores):
        raise SegmentationError("an instance id lacks a category or score")
    if set(res.categories) != set(res.scores):
        raise SegmentationError("categories and scores must have the same ids")
    if any(not (0.0 <= s <= 1.0) for s in res.scores.values()):
        raise SegmentationError("scores must lie in [0, 1]")
    if res.calib_version != calib_version:
        raise SegmentationError(f"calib_version {res.calib_version} != frame {calib_version}")
