"""Material prediction type (PRD 7.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Fixed class order (PRD 7.3); index == material_id.
MATERIAL_CLASSES: tuple[str, ...] = (
    "paper", "cloth", "wood", "plastic", "metal", "glass", "ceramic", "other_nonflammable",
)  # fmt: skip


@dataclass(frozen=True, eq=False)
class MaterialPrediction:
    """Distribution over materials, plus provenance and a review flag.

    ``class_probabilities`` f32 (n_materials,) summing to 1 (tol 1e-4), order MATERIAL_CLASSES.
    """

    class_probabilities: np.ndarray
    category_hint: str | None
    confidence: float
    source: Literal["model", "category_prior", "manual_override"]
    requires_review: bool

    def __post_init__(self) -> None:
        p = self.class_probabilities
        if p.dtype != np.float32 or p.ndim != 1:
            raise ValueError(f"class_probabilities must be float32 1-D, got {p.dtype} {p.shape}")
        if (p < 0).any() or not np.isfinite(p).all():
            raise ValueError("class_probabilities must be finite and non-negative")
        if abs(float(p.sum()) - 1.0) > 1e-4:
            raise ValueError(f"class_probabilities must sum to 1, got {float(p.sum())}")
        if self.source not in ("model", "category_prior", "manual_override"):
            raise ValueError(f"invalid source {self.source!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")

    @property
    def material_id(self) -> int:
        """Argmax class index."""
        return int(np.argmax(self.class_probabilities))
