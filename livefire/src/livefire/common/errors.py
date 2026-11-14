"""Shared exception types named by the interface contracts (PRD 9)."""

from __future__ import annotations


class CalibVersionError(ValueError):
    """A frame/camera/recording calib_version does not match the snapshot's."""


class DepthError(RuntimeError):
    """Depth model failure."""


class SegmentationError(RuntimeError):
    """Segmentation model failure."""


class MaterialError(RuntimeError):
    """Material model failure."""


class ReconstructionError(RuntimeError):
    """Scene reconstruction failed (e.g. insufficient or invalid depth)."""
