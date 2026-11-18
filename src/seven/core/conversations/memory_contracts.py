"""Shared data contracts for memory operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from seven.core.conversations.memory_types import (
    normalize_memory_category,
)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """Represent a candidate extracted from a user message."""

    content: str
    category: str
    importance: float
    confidence: float = field(default=1.0, compare=False)
    classification_confidence: float = field(
        default=1.0,
        compare=False,
    )
    classification_evidence: tuple[str, ...] = field(
        default=(),
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate classification metadata."""
        for name, value in (
            ("confidence", self.confidence),
            ("classification_confidence", self.classification_confidence),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(f"{name} must be numeric.")

            normalized = float(value)

            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite.")

            if not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0.")

            object.__setattr__(self, name, normalized)

        if not isinstance(self.classification_evidence, (tuple, list)):
            raise TypeError(
                "Classification evidence must be a tuple or list."
            )

        evidence: list[str] = []

        for item in self.classification_evidence:
            if not isinstance(item, str):
                raise TypeError("Classification evidence must contain strings.")

            normalized_item = item.strip()

            if normalized_item and normalized_item not in evidence:
                evidence.append(normalized_item)

        object.__setattr__(
            self,
            "classification_evidence",
            tuple(evidence),
        )


@dataclass(frozen=True, slots=True)
class MemoryClassification:
    """Represent a memory classification with confidence and evidence."""

    category: str
    confidence: float
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the classification result."""
        object.__setattr__(
            self,
            "category",
            normalize_memory_category(self.category),
        )

        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
        ):
            raise TypeError("Classification confidence must be numeric.")

        confidence = float(self.confidence)

        if not math.isfinite(confidence):
            raise ValueError("Classification confidence must be finite.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Classification confidence must be between 0.0 and 1.0."
            )

        object.__setattr__(self, "confidence", confidence)

        if not isinstance(self.evidence, (tuple, list)):
            raise TypeError("Classification evidence must be a tuple or list.")

        normalized_evidence: list[str] = []

        for item in self.evidence:
            if not isinstance(item, str):
                raise TypeError(
                    "Classification evidence must contain strings."
                )

            normalized_item = item.strip()

            if normalized_item and normalized_item not in normalized_evidence:
                normalized_evidence.append(normalized_item)

        object.__setattr__(
            self,
            "evidence",
            tuple(normalized_evidence),
        )


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    """Validated parameters for a memory search operation."""

    query: str
    limit: int = 5
    min_importance: float = 0.0
    categories: tuple[str, ...] = ()
    conversation_id: str | None = None
    include_global: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("Memory search query must be a string.")

        normalized_query = self.query.strip()

        if not normalized_query:
            raise ValueError("Memory search query cannot be empty.")

        object.__setattr__(self, "query", normalized_query)

        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("Memory search limit must be an integer.")

        if self.limit < 1:
            raise ValueError("Memory search limit must be positive.")

        if (
            isinstance(self.min_importance, bool)
            or not isinstance(self.min_importance, (int, float))
        ):
            raise TypeError("Minimum importance must be numeric.")

        importance = float(self.min_importance)

        if not math.isfinite(importance):
            raise ValueError("Minimum importance must be finite.")

        if not 0.0 <= importance <= 1.0:
            raise ValueError(
                "Minimum importance must be between 0.0 and 1.0."
            )

        object.__setattr__(self, "min_importance", importance)

        if not isinstance(self.categories, (tuple, list)):
            raise TypeError("Memory categories must be a tuple or list.")

        normalized_categories: list[str] = []

        for category in self.categories:
            normalized = normalize_memory_category(category)

            if normalized not in normalized_categories:
                normalized_categories.append(normalized)

        object.__setattr__(
            self,
            "categories",
            tuple(normalized_categories),
        )

        if self.conversation_id is not None:
            if not isinstance(self.conversation_id, str):
                raise TypeError("Conversation ID must be a string.")

            conversation_id = self.conversation_id.strip()

            if not conversation_id:
                raise ValueError("Conversation ID cannot be empty.")

            object.__setattr__(
                self,
                "conversation_id",
                conversation_id,
            )

        if not isinstance(self.include_global, bool):
            raise TypeError("Include global must be a boolean.")


@dataclass(frozen=True, slots=True)
class MemorySearchMatch:
    """A memory search result with relevance metadata."""

    memory_id: str
    content: str
    category: str
    importance: float
    relevance_score: float
    combined_score: float
    conversation_id: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str):
            raise TypeError("Memory ID must be a string.")

        memory_id = self.memory_id.strip()

        if not memory_id:
            raise ValueError("Memory ID cannot be empty.")

        object.__setattr__(self, "memory_id", memory_id)

        if not isinstance(self.content, str):
            raise TypeError("Memory content must be a string.")

        content = self.content.strip()

        if not content:
            raise ValueError("Memory content cannot be empty.")

        object.__setattr__(self, "content", content)

        object.__setattr__(
            self,
            "category",
            normalize_memory_category(self.category),
        )

        self._validate_score("importance", self.importance)
        self._validate_score("relevance_score", self.relevance_score)
        self._validate_score("combined_score", self.combined_score)

        if self.conversation_id is not None:
            if not isinstance(self.conversation_id, str):
                raise TypeError("Conversation ID must be a string.")

            conversation_id = self.conversation_id.strip()

            if not conversation_id:
                raise ValueError("Conversation ID cannot be empty.")

            object.__setattr__(
                self,
                "conversation_id",
                conversation_id,
            )

        if (
            self.created_at is not None
            and not isinstance(self.created_at, datetime)
        ):
            raise TypeError("Created at must be a datetime value.")

    @staticmethod
    def _validate_score(name: str, value: float) -> None:
        """Validate a normalized memory score."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        score = float(value)

        if not math.isfinite(score):
            raise ValueError(f"{name} must be finite.")

        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0.")