
"""Normalization and deduplication of memory candidates and records."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import Protocol, TypeVar

from seven.core.conversations.memory_contracts import MemoryCandidate


class MemoryContent(Protocol):
    """Protocol for objects containing memory text."""

    content: str


MemoryRecord = TypeVar("MemoryRecord", bound=MemoryContent)


class MemoryCandidateNormalizer:
    """Normalize memory text and remove duplicate memory records."""

    @staticmethod
    def normalize_text(content: str) -> str:
        """Create a canonical representation of memory text."""
        if not isinstance(content, str):
            raise TypeError("Memory content must be a string.")

        normalized = unicodedata.normalize("NFKC", content)
        normalized = normalized.casefold()

        normalized = (
            normalized
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u02bc", "'")
        )

        tokens = re.findall(
            r"\w+(?:'\w+)*",
            normalized,
            flags=re.UNICODE,
        )

        return " ".join(tokens)

    @classmethod
    def fingerprint(cls, content: str) -> str:
        """Generate a stable SHA-256 fingerprint for normalized content."""
        normalized = cls.normalize_text(content)

        if not normalized:
            raise ValueError(
                "Cannot generate a fingerprint for empty content."
            )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @classmethod
    def are_duplicates(
        cls,
        first: str,
        second: str,
    ) -> bool:
        """Determine whether two text values are exact normalized matches."""
        first_normalized = cls.normalize_text(first)
        second_normalized = cls.normalize_text(second)

        if not first_normalized or not second_normalized:
            return False

        return first_normalized == second_normalized

    @classmethod
    def deduplicate_candidates(
        cls,
        candidates: Iterable[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        """Remove duplicate candidates while preserving their first position."""
        unique: dict[str, MemoryCandidate] = {}

        for candidate in candidates:
            if not isinstance(candidate, MemoryCandidate):
                raise TypeError(
                    "All candidates must be MemoryCandidate instances."
                )

            normalized = cls.normalize_text(candidate.content)

            if not normalized:
                continue

            existing = unique.get(normalized)

            if existing is None:
                unique[normalized] = candidate
                continue

            candidate_rank = (
                candidate.confidence,
                candidate.importance,
            )
            existing_rank = (
                existing.confidence,
                existing.importance,
            )

            if candidate_rank > existing_rank:
                unique[normalized] = candidate

        return list(unique.values())

    @classmethod
    def deduplicate_memories(
        cls,
        memories: Iterable[MemoryRecord],
    ) -> tuple[MemoryRecord, ...]:
        """Remove duplicate memory records while preserving retrieval order."""
        unique: dict[str, MemoryRecord] = {}

        for memory in memories:
            content = getattr(memory, "content", None)

            if not isinstance(content, str):
                raise TypeError(
                    "Every memory must contain string content."
                )

            normalized = cls.normalize_text(content)

            if not normalized:
                continue

            if normalized not in unique:
                unique[normalized] = memory

        return tuple(unique.values())
