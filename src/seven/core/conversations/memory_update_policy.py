
"""Decision policy for updating existing memories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math


class MemoryUpdateAction(StrEnum):
    KEEP = "keep"
    UPDATE = "update"
    REINFORCE = "reinforce"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class MemoryUpdateDecision:
    """Represent the outcome of a memory update evaluation."""

    action: MemoryUpdateAction
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.action, MemoryUpdateAction):
            raise TypeError("Action must be a MemoryUpdateAction.")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Reason must be a non-empty string.")

        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence,
            (int, float),
        ):
            raise TypeError("Confidence must be numeric.")

        if not math.isfinite(self.confidence):
            raise ValueError("Confidence must be finite.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")


class MemoryUpdatePolicy:
    """Evaluate whether incoming information should update a memory."""

    def __init__(
        self,
        *,
        confidence_margin: float = 0.05,
        reinforcement_threshold: float = 0.05,
    ) -> None:
        """Initialize the update policy."""
        self._validate_threshold(
            confidence_margin,
            "Confidence margin",
        )
        self._validate_threshold(
            reinforcement_threshold,
            "Reinforcement threshold",
        )

        self._confidence_margin = confidence_margin
        self._reinforcement_threshold = reinforcement_threshold

    @property
    def confidence_margin(self) -> float:
        return self._confidence_margin

    @property
    def reinforcement_threshold(self) -> float:
        return self._reinforcement_threshold

    def evaluate(
        self,
        *,
        existing_content: str,
        incoming_content: str,
        existing_confidence: float,
        incoming_confidence: float,
    ) -> MemoryUpdateDecision:
        """Evaluate incoming information against an existing memory."""
        self._validate_content(existing_content, "Existing content")
        self._validate_content(incoming_content, "Incoming content")
        self._validate_confidence(
            existing_confidence,
            "Existing confidence",
        )
        self._validate_confidence(
            incoming_confidence,
            "Incoming confidence",
        )

        existing_normalized = self._normalize(existing_content)
        incoming_normalized = self._normalize(incoming_content)

        if existing_normalized == incoming_normalized:
            return MemoryUpdateDecision(
                action=MemoryUpdateAction.REINFORCE,
                reason="Incoming information confirms an existing memory.",
                confidence=max(
                    existing_confidence,
                    incoming_confidence,
                ),
            )

        confidence_difference = (
            incoming_confidence - existing_confidence
        )

        if confidence_difference >= self._confidence_margin:
            return MemoryUpdateDecision(
                action=MemoryUpdateAction.UPDATE,
                reason=(
                    "Incoming information has sufficiently higher "
                    "confidence."
                ),
                confidence=incoming_confidence,
            )

        if confidence_difference <= -self._confidence_margin:
            return MemoryUpdateDecision(
                action=MemoryUpdateAction.KEEP,
                reason=(
                    "Existing memory has sufficiently higher confidence."
                ),
                confidence=existing_confidence,
            )

        if (
            abs(confidence_difference)
            < self._reinforcement_threshold
        ):
            return MemoryUpdateDecision(
                action=MemoryUpdateAction.REVIEW,
                reason=(
                    "Conflicting information has similar confidence "
                    "and requires review."
                ),
                confidence=max(
                    existing_confidence,
                    incoming_confidence,
                ),
            )

        return MemoryUpdateDecision(
            action=MemoryUpdateAction.REVIEW,
            reason=(
                "Conflicting information does not meet the update "
                "confidence margin."
            ),
            confidence=max(
                existing_confidence,
                incoming_confidence,
            ),
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _validate_content(value: str, field: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string.")

        if not value.strip():
            raise ValueError(f"{field} must not be empty.")

    @staticmethod
    def _validate_confidence(value: float, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite.")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be between 0 and 1.")

    @staticmethod
    def _validate_threshold(value: float, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite.")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be between 0 and 1.")
