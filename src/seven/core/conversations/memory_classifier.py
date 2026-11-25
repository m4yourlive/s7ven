"""Rule-based classification of long-term memories."""

from __future__ import annotations

import re

from seven.core.conversations.memory_category_validator import (
    MemoryCategoryValidator,
)
from seven.core.conversations.memory_contracts import MemoryClassification
from seven.core.conversations.memory_taxonomy import (
    MemoryCategoryTaxonomy,
)


class MemoryClassifier:
    """Classify memory content and provide supporting evidence."""

    _CATEGORY_SIGNALS = {
        "goal": (
            "my long-term goal",
            "my goal",
            "my target",
            "goal",
            "target",
            "i want",
            "i aim",
            "i plan",
            "i intend",
            "i hope",
        ),
        "preference": (
            "my preference",
            "i prefer",
            "i like",
            "i dislike",
            "i love",
            "i hate",
            "i would rather",
            "favorite",
            "favourite",
        ),
        "knowledge": (
            "i have experience",
            "i am familiar with",
            "i specialize in",
            "my expertise",
            "my knowledge",
            "i know",
            "i learned",
            "i understand",
            "experience",
            "expertise",
            "knowledge",
        ),
        "project": (
            "my project",
            "project",
            "building",
            "developing",
            "creating",
            "development",
            "implementation",
        ),
        "education": (
            "i am studying",
            "i study",
            "studying",
            "school",
            "university",
            "college",
            "learning",
            "student",
            "education",
        ),
        "work": (
            "my job",
            "my career",
            "my profession",
            "i work",
            "occupation",
            "profession",
            "career",
            "job",
        ),
        "personal": (
            "my name",
            "my age",
            "my family",
            "i am",
            "i'm",
            "i use",
            "i live",
            "i have",
        ),
    }

    def __init__(
        self,
        category_validator: MemoryCategoryValidator | None = None,
    ) -> None:
        """Initialize the classifier with category validation."""
        if category_validator is not None and not isinstance(
            category_validator,
            MemoryCategoryValidator,
        ):
            raise TypeError(
                "category_validator must be a MemoryCategoryValidator."
            )

        self._category_validator = (
            category_validator or MemoryCategoryValidator()
        )

    def classify(self, content: str) -> str:
        """Return the most relevant category for memory content."""
        return self.classify_with_evidence(content).category

    def classify_with_evidence(
        self,
        content: str,
    ) -> MemoryClassification:
        """Classify content and return matching evidence."""
        if not isinstance(content, str):
            raise TypeError("Memory content must be a string.")

        normalized = content.strip().casefold()

        if not normalized:
            raise ValueError("Memory content cannot be empty.")

        for category in MemoryCategoryTaxonomy.classification_order():
            signals = self._CATEGORY_SIGNALS[category]

            evidence = tuple(
                signal
                for signal in signals
                if self._contains_phrase(normalized, signal)
            )

            if evidence:
                confidence = min(
                    0.98,
                    0.82 + 0.04 * (len(evidence) - 1),
                )

                resolved_category = self._category_validator.validate(
                    category
                )

                return MemoryClassification(
                    category=resolved_category,
                    confidence=confidence,
                    evidence=evidence,
                )

        fallback = self._category_validator.resolve(
            "__unclassified__"
        )

        return MemoryClassification(
            category=fallback,
            confidence=0.35,
            evidence=(),
        )

    @staticmethod
    def _contains_phrase(content: str, phrase: str) -> bool:
        """Match complete phrases without matching partial words."""
        pattern = rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)"

        return re.search(pattern, content) is not None