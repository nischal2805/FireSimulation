"""Warp device selection and host->device upload helpers (ADR-0001)."""

from __future__ import annotations

import numpy as np
import warp as wp

from livefire.common.config import load_yaml

wp.config.quiet = True
# gas.py scatters with float atomic_add, whose summation order is nondeterministic on CUDA. Warp's RUN_TO_RUN mode
# makes such atomics reproducible between runs on the same GPU/driver/build. Must be set before any module is
# created. GPU replay relies on it and must be confirmed on the RTX 4060 (ADR-0002); CPU is serial and exact.
wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
_initialised = False


def select_device(name: str | None = None) -> str:
    """Return a validated Warp device string. ``name`` defaults to ``configs/app.yaml: device``
    ("cpu" here, "cuda:0" on the GPU machine). Initialises Warp once; raises if the device is absent."""
    global _initialised
    if not _initialised:
        wp.init()
        _initialised = True
    dev = name if name is not None else str(load_yaml("app").get("device", "cpu"))
    wp.get_device(dev)  # raises if unavailable
    return dev


def to_device(a: np.ndarray, dtype: type, device: str) -> wp.array:
    """Copy host array ``a`` to a new Warp array of ``dtype`` on ``device`` (caller owns it)."""
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device=device)
