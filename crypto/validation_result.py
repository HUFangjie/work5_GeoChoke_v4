from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PipelineValidationResult:
    profile_id: str
    number_of_vectors: int
    weights: Sequence[float]
    mse: float
    max_abs_error: float
    relative_l2_error: float
    norm_ratio: float
    passed: bool

    def as_row(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "number_of_vectors": self.number_of_vectors,
            "weights": list(self.weights),
            "aggregate_mse": self.mse,
            "aggregate_max_abs_error": self.max_abs_error,
            "relative_l2_error": self.relative_l2_error,
            "norm_ratio": self.norm_ratio,
            "passed": self.passed,
        }
