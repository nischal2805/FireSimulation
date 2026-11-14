"""Degradation controller (PRD 5.6): counts consecutive slow frames and walks the ladder."""

from __future__ import annotations

from livefire.app.hud import DegradationLevel


class DegradationController:
    """rt factor < ``threshold`` for ``frames`` consecutive frames -> next ladder level (counter then resets).
    Pure logic, no rendering; the caller applies and logs the returned level."""

    def __init__(self, ladder: tuple[DegradationLevel, ...], threshold: float, frames: int) -> None:
        self.ladder, self.threshold, self.frames = ladder, threshold, frames
        self.level = 0
        self.low = 0

    def update(self, rt_factor: float) -> int | None:
        """Feed one frame's rt factor; returns the new level if the ladder advanced, else None."""
        self.low = self.low + 1 if rt_factor < self.threshold else 0
        if self.low >= self.frames and self.level < len(self.ladder) - 1:
            self.level += 1
            self.low = 0
            return self.level
        return None
