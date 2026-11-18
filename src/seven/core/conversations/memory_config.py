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
    retrieval_min_relevance: float = 0.0
    retrieval_min_combined_score: float = 0.0

    relevance_weight: float = 0.7
    importance_weight: float = 0.3

    lexical_coverage_weight: float = 0.55
    lexical_precision_weight: float = 0.15
    lexical_jaccard_weight: float = 0.15
    lexical_phrase_weight: float = 0.15

    max_context_memories: int = 5

    context_lexical_weight: float = 0.7
    context_category_weight: float = 0.1
    context_conversation_weight: float = 0.1
    context_recency_weight: float = 0.1

    recency_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        """Validate architecture configuration."""
        integer_fields = (
            "retrieval_limit",
            "retrieval_candidate_multiplier",
            "retrieval_min_candidates",
            "max_context_memories",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 1:
                raise ValueError(
                    f"{field_name} must be positive."
                )

        unit_interval_fields = (
            "retrieval_min_importance",
            "retrieval_min_relevance",
            "retrieval_min_combined_score",
            "relevance_weight",
            "importance_weight",
            "lexical_coverage_weight",
            "lexical_precision_weight",
            "lexical_jaccard_weight",
            "lexical_phrase_weight",
            "context_lexical_weight",
            "context_category_weight",
            "context_conversation_weight",
            "context_recency_weight",
        )

        for field_name in unit_interval_fields:
            self._validate_unit_interval(
                field_name,
                getattr(self, field_name),
            )

        self._validate_weight_total(
            "relevance and importance",
            (
                self.relevance_weight,
                self.importance_weight,
            ),
        )

        self._validate_weight_total(
            "lexical scoring",
            (
                self.lexical_coverage_weight,
                self.lexical_precision_weight,
                self.lexical_jaccard_weight,
                self.lexical_phrase_weight,
            ),
        )

        self._validate_weight_total(
            "context scoring",
            (
                self.context_lexical_weight,
                self.context_category_weight,
                self.context_conversation_weight,
                self.context_recency_weight,
            ),
        )

        if (
            isinstance(self.recency_half_life_days, bool)
            or not isinstance(
                self.recency_half_life_days,
                (int, float),
            )
        ):
            raise TypeError(
                "Recency half-life must be numeric."
            )

        half_life = float(self.recency_half_life_days)

        if not math.isfinite(half_life):
            raise ValueError(
                "Recency half-life must be finite."
            )

        if half_life <= 0.0:
            raise ValueError(
                "Recency half-life must be positive."
            )

        object.__setattr__(
            self,
            "recency_half_life_days",
            half_life,
        )

        for field_name in unit_interval_fields:
            object.__setattr__(
                self,
                field_name,
                float(getattr(self, field_name)),
            )

    @staticmethod
    def _validate_unit_interval(
        name: str,
        value: float,
    ) -> None:
        """Validate a finite numeric value between zero and one."""
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(f"{name} must be numeric.")

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")

        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0."
            )

    @staticmethod
    def _validate_weight_total(
        name: str,
        weights: tuple[float, ...],
    ) -> None:
        """Ensure a group of scoring weights sums to one."""
        total = sum(weights)

        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{name.capitalize()} weights must sum to 1.0."
            )

    def candidate_limit(
        self,
        requested_limit: int | None = None,
    ) -> int:
        """Calculate the number of candidates to fetch for retrieval."""
        limit = (
            self.retrieval_limit
            if requested_limit is None
            else requested_limit
        )

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(
                "Requested retrieval limit must be an integer."
            )

        if limit < 1:
            raise ValueError(
                "Requested retrieval limit must be positive."
            )

        return max(
            limit * self.retrieval_candidate_multiplier,
            self.retrieval_min_candidates,
        )