"""Monocular depth backends and table-plane anchoring (P1-004, ADR-0005). Imports without torch."""

from livefire.perception.depth.anchor import (
    AnchorConfig,
    AnchoredDepthBackend,
    AnchorFit,
    TableRegion,
    apply_anchor,
    fit_anchor,
    table_plane_depth,
)
from livefire.perception.depth.base import RawDepth
from livefire.perception.depth.oracle import OracleDepthBackend

__all__ = [
    "AnchorConfig", "AnchorFit", "AnchoredDepthBackend", "OracleDepthBackend", "RawDepth", "TableRegion",
    "apply_anchor", "fit_anchor", "table_plane_depth",
]  # fmt: skip
