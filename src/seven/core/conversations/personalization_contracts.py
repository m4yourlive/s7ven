
"""Data contracts for the personalization engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from seven.core.conversations.user_profile_contracts import UserProfile


class PersonalizationCategory(StrEnum):
    """Define supported personalization dimensions."""

    COMMUNICATION_STYLE = "communication_style"
    RESPONSE_LENGTH = "response_length"
    FORMATTING = "formatting"
    TECHNICAL_DEPTH = "technical_depth"
    LEARNING_STYLE = "learning_style"
    DOMAIN_CONTEXT = "domain_context"


def _normalize_text(value: str, field_name: str) -> str:
    """Normalize and validate non-empty text."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")

    return normalized


def _normalize_confidence(value: float) -> float:
    """Validate a confidence score between zero and one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Confidence must be numeric.")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError("Confidence must be finite.")

    if not 0.0 <= normalized <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")

    return normalized


def _normalize_priority(value: int) -> int:
    """Validate a non-negative priority value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Priority must be an integer.")

    if value < 0:
        raise ValueError("Priority cannot be negative.")

    return value


def _normalize_category(
    value: PersonalizationCategory | str,
) -> PersonalizationCategory:
    """Normalize a personalization category."""
    if isinstance(value, PersonalizationCategory):
        return value

    if not isinstance(value, str):
        raise TypeError("Category must be a string or PersonalizationCategory.")

    try:
        return PersonalizationCategory(value.strip().casefold())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported personalization category: {value!r}."
        ) from exc


@dataclass(frozen=True, slots=True)
class PersonalizationDirective:
    """Represent one actionable personalization instruction."""

    category: PersonalizationCategory
    instruction: str
    confidence: float = 1.0
    priority: int = 50
    source: str = "profile"
    key: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate a personalization directive."""
        object.__setattr__(
            self,
            "category",
            _normalize_category(self.category),
        )
        object.__setattr__(
            self,
            "instruction",
            _normalize_text(self.instruction, "Instruction"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "priority",
            _normalize_priority(self.priority),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_text(self.source, "Source").casefold(),
        )

        if self.key is not None:
            object.__setattr__(
                self,
                "key",
                _normalize_text(self.key, "Directive key").casefold(),
            )


@dataclass(frozen=True, slots=True)
class PersonalizationRequest:
    """Represent the input supplied to the personalization engine."""

    user_message: str
    profile: UserProfile | None = None
    memory_context: str = ""
    conversation_context: str = ""
    min_confidence: float = 0.0
    max_directives: int = 10

    def __post_init__(self) -> None:
        """Validate personalization request data."""
        object.__setattr__(
            self,
            "user_message",
            _normalize_text(self.user_message, "User message"),
        )

        if self.profile is not None and not isinstance(
            self.profile,
            UserProfile,
        ):
            raise TypeError("Profile must be a UserProfile or None.")

        for field_name in ("memory_context", "conversation_context"):
            value = getattr(self, field_name)

            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string.")

            object.__setattr__(self, field_name, value.strip())

        object.__setattr__(
            self,
            "min_confidence",
            _normalize_confidence(self.min_confidence),
        )

        if isinstance(self.max_directives, bool) or not isinstance(
            self.max_directives,
            int,
        ):
            raise TypeError("Maximum directives must be an integer.")

        if self.max_directives < 1:
            raise ValueError("Maximum directives must be at least 1.")


@dataclass(frozen=True, slots=True)
class PersonalizationResult:
    """Represent the output produced by the personalization engine."""

    directives: tuple[PersonalizationDirective, ...] = ()
    context: str = ""
    applied: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize personalization results."""
        if not isinstance(self.directives, (tuple, list)):
            raise TypeError("Directives must be a tuple or list.")

        directives = tuple(self.directives)

        if not all(
            isinstance(item, PersonalizationDirective)
            for item in directives
        ):
            raise TypeError(
                "Directives must contain PersonalizationDirective values."
            )

        if not isinstance(self.context, str):
            raise TypeError("Personalization context must be a string.")

        if not isinstance(self.applied, bool):
            raise TypeError("Applied must be a boolean.")

        if self.applied and not directives:
            raise ValueError(
                "A personalization result must contain directives "
                "when applied is true."
            )

        object.__setattr__(self, "directives", directives)
        object.__setattr__(self, "context", self.context.strip())
