"""Shared configuration for the memory management architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryArchitectureConfig:
    """Central configuration for memory retrieval and context injection."""

    retrieval_limit: int = 5
    retrieval_candidate_multiplier: int = 5
    retrieval_min_candidates: int = 20
    retrieval_min_importance: float = 0.0
    relevance_weight: float = 0.7
    importance_weight: float = 0.3
    max_context_memories: int = 5

    def __post_init__(self) -> None:
        integer_fields = (
            "retrieval_limit",
            "retrieval_candidate_multiplier",
            "retrieval_min_candidates",
            "max_context_memories",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")

            if value < 1:
                raise ValueError(f"{field_name} must be positive.")

        self._validate_unit_interval(
            "retrieval_min_importance",
            self.retrieval_min_importance,
        )
        self._validate_unit_interval(
            "relevance_weight",
            self.relevance_weight,
        )
        self._validate_unit_interval(
            "importance_weight",
            self.importance_weight,
        )

        total_weight = (
            float(self.relevance_weight)
            + float(self.importance_weight)
        )

        if not math.isclose(
            total_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Relevance and importance weights must sum to 1.0."
            )

        object.__setattr__(
            self,
            "retrieval_min_importance",
            float(self.retrieval_min_importance),
        )
        object.__setattr__(
            self,
            "relevance_weight",
            float(self.relevance_weight),
        )
        object.__setattr__(
            self,
            "importance_weight",
            float(self.importance_weight),
        )

    @staticmethod
    def _validate_unit_interval(name: str, value: float) -> None:
        """Validate a finite numeric value between zero and one."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")

        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0."
            )

    def candidate_limit(self, requested_limit: int | None = None) -> int:
        """Calculate the number of candidates to fetch for retrieval."""
        limit = (
            self.retrieval_limit
            if requested_limit is None
            else requested_limit
        )

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("Requested retrieval limit must be an integer.")

        if limit < 1:
            raise ValueError(
                "Requested retrieval limit must be positive."
            )

        return max(
            limit * self.retrieval_candidate_multiplier,
            self.retrieval_min_candidates,
        )