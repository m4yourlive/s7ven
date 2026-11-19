
"""Detect potential conflicts between existing and incoming memories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class MemoryConflictType(StrEnum):
    NONE = "none"
    DIRECT_CONTRADICTION = "direct_contradiction"
    VALUE_CHANGE = "value_change"
    POSSIBLE_CONFLICT = "possible_conflict"


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """Describe a detected conflict between two memories."""

    conflict_type: MemoryConflictType
    is_conflict: bool
    shared_terms: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.conflict_type, MemoryConflictType):
            raise TypeError("Conflict type must be a MemoryConflictType.")

        if not isinstance(self.is_conflict, bool):
            raise TypeError("is_conflict must be a boolean.")

        if not isinstance(self.shared_terms, tuple):
            raise TypeError("Shared terms must be a tuple.")

        if any(
            not isinstance(term, str) or not term.strip()
            for term in self.shared_terms
        ):
            raise ValueError("Shared terms must contain non-empty strings.")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Reason must be a non-empty string.")

        if self.is_conflict != (
            self.conflict_type != MemoryConflictType.NONE
        ):
            raise ValueError("Conflict status must match the conflict type.")


class MemoryConflictDetector:
    """Detect lexical contradictions and potential memory conflicts."""

    _STOP_WORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "as",
            "at",
            "be",
            "for",
            "from",
            "i",
            "in",
            "is",
            "it",
            "my",
            "of",
            "on",
            "or",
            "the",
            "to",
            "was",
            "with",
        }
    )

    _OPPOSITES = (
        frozenset({"like", "dislike"}),
        frozenset({"likes", "dislikes"}),
        frozenset({"love", "hate"}),
        frozenset({"loves", "hates"}),
        frozenset({"prefer", "avoid"}),
        frozenset({"prefers", "avoids"}),
        frozenset({"dark", "light"}),
        frozenset({"high", "low"}),
        frozenset({"increase", "decrease"}),
        frozenset({"yes", "no"}),
        frozenset({"true", "false"}),
        frozenset({"enabled", "disabled"}),
        frozenset({"active", "inactive"}),
        frozenset({"old", "new"}),
    )

    _NEGATIONS = frozenset(
        {
            "not",
            "never",
            "no",
            "cannot",
            "can't",
            "dont",
            "don't",
            "doesnt",
            "doesn't",
            "isnt",
            "isn't",
        }
    )

    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

    def __init__(
        self,
        *,
        minimum_shared_terms: int = 1,
        minimum_overlap: float = 0.3,
    ) -> None:
        """Initialize conflict detection thresholds."""
        if isinstance(minimum_shared_terms, bool) or not isinstance(
            minimum_shared_terms,
            int,
        ):
            raise TypeError("Minimum shared terms must be an integer.")

        if minimum_shared_terms < 1:
            raise ValueError("Minimum shared terms must be at least 1.")

        if isinstance(minimum_overlap, bool) or not isinstance(
            minimum_overlap,
            (int, float),
        ):
            raise TypeError("Minimum overlap must be numeric.")

        if not 0.0 <= minimum_overlap <= 1.0:
            raise ValueError("Minimum overlap must be between 0 and 1.")

        self._minimum_shared_terms = minimum_shared_terms
        self._minimum_overlap = float(minimum_overlap)

    @property
    def minimum_shared_terms(self) -> int:
        return self._minimum_shared_terms

    @property
    def minimum_overlap(self) -> float:
        return self._minimum_overlap

    def detect(
        self,
        existing_content: str,
        incoming_content: str,
    ) -> MemoryConflict:
        """Detect whether incoming content conflicts with existing content."""
        self._validate_content(existing_content, "Existing content")
        self._validate_content(incoming_content, "Incoming content")

        existing_tokens = self._tokenize(existing_content)
        incoming_tokens = self._tokenize(incoming_content)

        if existing_tokens == incoming_tokens:
            return self._no_conflict(
                "Both memories contain equivalent normalized content."
            )

        existing_terms = self._meaningful_terms(existing_tokens)
        incoming_terms = self._meaningful_terms(incoming_tokens)

        shared_terms = tuple(
            sorted(existing_terms.intersection(incoming_terms))
        )

        if len(shared_terms) < self._minimum_shared_terms:
            return self._no_conflict(
                "There is insufficient shared context."
            )

        overlap = len(shared_terms) / max(
            len(existing_terms),
            len(incoming_terms),
            1,
        )

        if overlap < self._minimum_overlap:
            return self._no_conflict(
                "The shared context is below the conflict threshold."
            )

        if self._has_value_change(
            existing_tokens,
            incoming_tokens,
        ):
            return MemoryConflict(
                conflict_type=MemoryConflictType.VALUE_CHANGE,
                is_conflict=True,
                shared_terms=shared_terms,
                reason=(
                    "The memories share context but contain different "
                    "numeric values."
                ),
            )

        if (
            self._has_opposite_terms(
                existing_tokens,
                incoming_tokens,
            )
            or self._has_negation_difference(
                existing_tokens,
                incoming_tokens,
            )
        ):
            return MemoryConflict(
                conflict_type=MemoryConflictType.DIRECT_CONTRADICTION,
                is_conflict=True,
                shared_terms=shared_terms,
                reason=(
                    "The memories contain opposing terms or different "
                    "negation signals."
                ),
            )

        return MemoryConflict(
            conflict_type=MemoryConflictType.POSSIBLE_CONFLICT,
            is_conflict=True,
            shared_terms=shared_terms,
            reason=(
                "The memories share substantial context but contain "
                "different information."
            ),
        )

    @classmethod
    def _tokenize(cls, content: str) -> tuple[str, ...]:
        normalized = content.casefold().replace("’", "'")
        return tuple(cls._TOKEN_PATTERN.findall(normalized))

    @classmethod
    def _meaningful_terms(
        cls,
        tokens: tuple[str, ...],
    ) -> set[str]:
        return {
            token
            for token in tokens
            if token not in cls._STOP_WORDS
            and token not in cls._NEGATIONS
        }

    @staticmethod
    def _has_value_change(
        existing_tokens: tuple[str, ...],
        incoming_tokens: tuple[str, ...],
    ) -> bool:
        existing_numbers = {
            token for token in existing_tokens if token.isdigit()
        }
        incoming_numbers = {
            token for token in incoming_tokens if token.isdigit()
        }

        return bool(
            existing_numbers
            and incoming_numbers
            and existing_numbers != incoming_numbers
        )

    @classmethod
    def _has_opposite_terms(
        cls,
        existing_tokens: tuple[str, ...],
        incoming_tokens: tuple[str, ...],
    ) -> bool:
        existing_set = set(existing_tokens)
        incoming_set = set(incoming_tokens)

        return any(
            bool(pair.intersection(existing_set))
            and bool(pair.intersection(incoming_set))
            and pair.intersection(existing_set)
            != pair.intersection(incoming_set)
            for pair in cls._OPPOSITES
        )

    @classmethod
    def _has_negation_difference(
        cls,
        existing_tokens: tuple[str, ...],
        incoming_tokens: tuple[str, ...],
    ) -> bool:
        existing_negations = set(existing_tokens).intersection(
            cls._NEGATIONS
        )
        incoming_negations = set(incoming_tokens).intersection(
            cls._NEGATIONS
        )

        return existing_negations != incoming_negations

    @staticmethod
    def _no_conflict(reason: str) -> MemoryConflict:
        return MemoryConflict(
            conflict_type=MemoryConflictType.NONE,
            is_conflict=False,
            shared_terms=(),
            reason=reason,
        )

    @staticmethod
    def _validate_content(value: str, field: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string.")

        if not value.strip():
            raise ValueError(f"{field} must not be empty.")
