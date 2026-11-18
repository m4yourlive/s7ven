"""Quality assessment for extracted memory candidates."""

from __future__ import annotations

import math
import re

from seven.core.conversations.memory_contracts import MemoryCandidate

class MemoryQualityValidator:
    """Assess whether extracted memory candidates contain reliable information."""

    _DIRECT_SIGNALS = (
        "i am",
        "i'm",
        "i use",
        "i prefer",
        "i like",
        "i dislike",
        "i love",
        "i hate",
        "i want",
        "i need",
        "i plan",
        "i intend",
        "i aim",
        "i study",
        "i work",
        "i have",
        "my project",
        "my goal",
        "my preference",
        "my name",
        "my age",
        "remember that",
        "from now on",
        "going forward",
    )

    _DIRECTIVE_SIGNALS = (
        "remember that",
        "from now on",
        "going forward",
    )

    _UNCERTAINTY_SIGNALS = (
        "maybe",
        "perhaps",
        "probably",
        "i think",
        "i guess",
        "i might",
        "i may",
        "i could",
        "i wish",
    )

    _VAGUE_ENDINGS = {
        "it",
        "this",
        "that",
        "them",
        "stuff",
        "things",
        "something",
        "anything",
    }

    _KNOWN_CATEGORIES = {
        "personal",
        "preference",
        "project",
        "goal",
        "work",
        "education",
        "knowledge",
    }

    def __init__(self, min_confidence: float = 0.65) -> None:
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
        ):
            raise TypeError("Minimum confidence must be numeric.")

        normalized = float(min_confidence)

        if not math.isfinite(normalized):
            raise ValueError("Minimum confidence must be finite.")

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                "Minimum confidence must be between 0.0 and 1.0."
            )

        self._min_confidence = normalized

    @property
    def min_confidence(self) -> float:
        """Return the minimum accepted confidence score."""
        return self._min_confidence

    def assess(self, candidate: MemoryCandidate) -> float:
        """Calculate a normalized confidence score for a candidate."""
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError("Candidate must be a MemoryCandidate.")

        content = candidate.content.strip().casefold()
        category = candidate.category.strip().casefold()

        score = 0.30

        if self._contains_any(content, self._DIRECT_SIGNALS):
            score += 0.25

        if category in self._KNOWN_CATEGORIES:
            score += 0.15

        if len(content) >= 30:
            score += 0.10

        words = re.findall(r"\b[\w']+\b", content)

        if len(set(words)) >= 5:
            score += 0.10

        if self._contains_any(content, self._DIRECTIVE_SIGNALS):
            score += 0.10

        if content.endswith("?"):
            score -= 0.45

        if self._contains_any(content, self._UNCERTAINTY_SIGNALS):
            score -= 0.30

        if words and words[-1] in self._VAGUE_ENDINGS:
            score -= 0.30

        return round(max(0.0, min(score, 1.0)), 4)

    def is_acceptable(self, confidence: float) -> bool:
        """Check whether a confidence score meets the configured threshold."""
        if isinstance(confidence, bool) or not isinstance(
            confidence,
            (int, float),
        ):
            raise TypeError("Confidence must be numeric.")

        normalized = float(confidence)

        if not math.isfinite(normalized):
            raise ValueError("Confidence must be finite.")

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                "Confidence must be between 0.0 and 1.0."
            )

        return normalized >= self._min_confidence

    @staticmethod
    def _contains_any(content: str, signals: tuple[str, ...]) -> bool:
        """Match phrases without matching partial words."""
        for signal in signals:
            pattern = (
                rf"(?<!\w){re.escape(signal.casefold())}(?!\w)"
            )

            if re.search(pattern, content):
                return True

        return False