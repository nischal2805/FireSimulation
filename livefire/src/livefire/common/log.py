"""Logging setup; level from configs/app.yaml `log_level`."""

from __future__ import annotations

import logging

from livefire.common.config import load_yaml


def get_logger(name: str) -> logging.Logger:
    """Return a logger named `name` with level from app.yaml (handler added once per logger)."""
    logger = logging.getLogger(name)
    logger.setLevel(str(load_yaml("app").get("log_level", "INFO")).upper())
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    return logger
