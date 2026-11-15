"""Keyframe instance segmentation backends (P3-001, ADR-0007). Imports without torch."""

from livefire.perception.segment.base import (
    load_segment_config,
    result_from_masks,
    segment_extra_available,
    validate_segmentation_result,
)
from livefire.perception.segment.factory import BACKEND_NAMES, create_segmentation_backend
from livefire.perception.segment.oracle import OracleSegmentationBackend

__all__ = [
    "BACKEND_NAMES", "OracleSegmentationBackend", "create_segmentation_backend", "load_segment_config",
    "result_from_masks", "segment_extra_available", "validate_segmentation_result",
]  # fmt: skip
