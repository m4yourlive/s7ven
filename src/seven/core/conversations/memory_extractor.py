"""Automatic extraction of long-term memory candidates."""

from __future__ import annotations

import re

from seven.core.conversations.memory_classifier import MemoryClassifier
from seven.core.conversations.memory_contracts import MemoryCandidate
from seven.core.conversations.memory_normalizer import (
    MemoryCandidateNormalizer,
)
from seven.core.conversations.memory_quality import MemoryQualityValidator
from seven.core.conversations.memory_taxonomy import (
    MemoryCategoryTaxonomy,
)


class MemoryExtractor:
    """Extract potentially useful long-term memories from text."""

    _MEMORY_SIGNALS = (
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
        "i know",
        "i learned",
        "i understand",
        "i have experience",
        "i am familiar with",
        "i specialize in",
        "my expertise",
        "my knowledge",
        "my project",
        "my goal",
        "my long-term goal",
        "my preference",
        "my name",
        "my age",
        "my favorite",
        "my favourite",
        "remember that",
        "from now on",
        "going forward",
    )

    _DIRECTIVE_SIGNALS = (
        "remember that",
        "from now on",
        "going forward",
    )

    def __init__(
        self,
        min_length: int = 12,
        max_length: int = 500,
        quality_validator: MemoryQualityValidator | None = None,
        classifier: MemoryClassifier | None = None,
    ) -> None:
        if isinstance(min_length, bool) or not isinstance(min_length, int):
            raise TypeError("Minimum length must be an integer.")

        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("Maximum length must be an integer.")

        if min_length < 1:
            raise ValueError("Minimum length must be positive.")

        if max_length < min_length:
            raise ValueError(
                "Maximum length must be greater than or equal "
                "to minimum length."
            )

        if quality_validator is not None and not isinstance(
            quality_validator,
            MemoryQualityValidator,
        ):
            raise TypeError(
                "quality_validator must be a MemoryQualityValidator."
            )

        if classifier is not None and not isinstance(
            classifier,
            MemoryClassifier,
        ):
            raise TypeError("classifier must be a MemoryClassifier.")

        self._min_length = min_length
        self._max_length = max_length
        self._quality_validator = (
            quality_validator or MemoryQualityValidator()
        )
        self._classifier = classifier or MemoryClassifier()

    def extract(self, text: str) -> list[MemoryCandidate]:
        """Extract unique and quality-validated memory candidates."""
        if not isinstance(text, str):
            raise TypeError("Memory extraction input must be a string.")

        if not text.strip():
            return []

        candidates: list[MemoryCandidate] = []

        for sentence in self._split_sentences(text):
            content = self._normalize_content(sentence)

            if not self._is_valid_candidate(content):
                continue

            if not self._contains_memory_signal(content):
                continue

            classification = self._classifier.classify_with_evidence(
                content
            )

            category = classification.category
            importance = self._calculate_importance(
                content,
                category,
            )

            candidate = MemoryCandidate(
                content=content,
                category=category,
                importance=importance,
                classification_confidence=classification.confidence,
                classification_evidence=classification.evidence,
            )

            confidence = self._quality_validator.assess(candidate)

            if not self._quality_validator.is_acceptable(confidence):
                continue

            candidates.append(
                MemoryCandidate(
                    content=content,
                    category=category,
                    importance=importance,
                    confidence=confidence,
                    classification_confidence=classification.confidence,
                    classification_evidence=classification.evidence,
                )
            )

        return MemoryCandidateNormalizer.deduplicate_candidates(
            candidates
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into independent sentence candidates."""
        return re.split(r"(?<=[.!?])\s+|\n+", text.strip())

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Normalize whitespace and surrounding punctuation."""
        return re.sub(
            r"\s+",
            " ",
            content.strip().strip(" \t\r\n.!"),
        )

    @staticmethod
    def _normalize_for_comparison(content: str) -> str:
        """Create a normalized representation for deduplication."""
        return MemoryCandidateNormalizer.normalize_text(content)

    @staticmethod
    def _contains_phrase(content: str, phrase: str) -> bool:
        """Match a phrase without matching partial words."""
        pattern = rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)"

        return re.search(pattern, content.casefold()) is not None

    def _is_valid_candidate(self, content: str) -> bool:
        """Check whether a candidate satisfies length constraints."""
        return self._min_length <= len(content) <= self._max_length

    def _contains_memory_signal(self, content: str) -> bool:
        """Determine whether a sentence contains a memory signal."""
        return any(
            self._contains_phrase(content, signal)
            for signal in self._MEMORY_SIGNALS
        )

    def _classify_category(self, content: str) -> str:
        """Classify a candidate through the dedicated classifier."""
        return self._classifier.classify(content)

    @staticmethod
    def _calculate_importance(
        content: str,
        category: str,
    ) -> float:
        """Calculate importance using centralized category metadata."""
        importance = MemoryCategoryTaxonomy.get_importance(category)

        if any(
            MemoryExtractor._contains_phrase(content, signal)
            for signal in MemoryExtractor._DIRECTIVE_SIGNALS
        ):
            importance += 0.10

        return min(importance, 1.0)