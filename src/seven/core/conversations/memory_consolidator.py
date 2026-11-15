
"""Extract and consolidate persistent memories from conversation messages."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.message_service import MessageService
from seven.core.database.models import Memory, Message


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """Represent a candidate extracted from a user message."""

    content: str
    category: str
    importance: float


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
    "i plan",
    "i intend",
    "i study",
    "i work",
    "my project",
    "my goal",
    "my long-term goal",
    "my name",
    "my age",
    "from now on",
    "remember that",
    "going forward",
)

_CATEGORY_KEYWORDS = {
    "project": (
        "project",
        "building",
        "developing",
        "creating",
        "develop",
        "implementation",
    ),
    "preference": (
        "prefer",
        "preference",
        "i like",
        "i dislike",
        "i love",
        "i hate",
        "i would rather",
    ),
    "goal": (
        "goal",
        "target",
        "i want",
        "i aim",
        "i plan",
        "i intend",
        "i hope",
    ),
    "work": (
        "work",
        "job",
        "career",
        "profession",
        "occupation",
    ),
    "education": (
        "study",
        "studying",
        "school",
        "university",
        "college",
        "learning",
        "student",
    ),
    "personal": (
        "i am",
        "i'm",
        "my name",
        "my age",
        "i use",
        "i live",
        "my family",
    ),
}

_CATEGORY_PRIORITY = (
    "goal",
    "preference",
    "project",
    "education",
    "work",
    "personal",
)


class MemoryConsolidator:
    """Extract durable information from conversations and store it as memories."""

    def __init__(
        self,
        message_service: MessageService,
        memory_service: MemoryService,
    ) -> None:
        """Initialize the memory consolidator."""
        if not isinstance(message_service, MessageService):
            raise TypeError("message_service must be a MessageService instance.")

        if not isinstance(memory_service, MemoryService):
            raise TypeError("memory_service must be a MemoryService instance.")

        self.message_service = message_service
        self.memory_service = memory_service

    async def consolidate(
        self,
        conversation_id: str,
    ) -> list[Memory]:
        """Extract and persist new memories from a conversation."""
        messages = await self.message_service.get_messages(conversation_id)
        candidates = self.extract_candidates(messages)

        consolidated: list[Memory] = []

        for candidate in candidates:
            existing_memories = await self.memory_service.search(
                candidate.content,
                limit=20,
            )

            if self._is_duplicate(
                candidate.content,
                existing_memories,
            ):
                continue

            memory = await self.memory_service.create(
                conversation_id=conversation_id,
                content=candidate.content,
                category=candidate.category,
                importance=candidate.importance,
            )

            consolidated.append(memory)

        return consolidated

    def extract_candidates(
        self,
        messages: Sequence[Message],
    ) -> list[MemoryCandidate]:
        """Extract potential memories from user messages."""
        candidates: list[MemoryCandidate] = []
        seen: set[str] = set()

        for message in messages:
            if message.role != "user":
                continue

            for sentence in self._split_sentences(message.content):
                candidate = self._build_candidate(sentence)

                if candidate is None:
                    continue

                normalized = self._normalize(candidate.content)

                if normalized in seen:
                    continue

                seen.add(normalized)
                candidates.append(candidate)

        return candidates

    def _build_candidate(
        self,
        sentence: str,
    ) -> MemoryCandidate | None:
        """Build a memory candidate from a sentence."""
        content = sentence.strip()

        if len(content) < 12 or len(content) > 500:
            return None

        normalized = self._normalize(content)

        if not any(
            signal in normalized
            for signal in _MEMORY_SIGNALS
        ):
            return None

        category = self._categorize(normalized)
        importance = self._calculate_importance(
            category=category,
            content=normalized,
        )

        return MemoryCandidate(
            content=content,
            category=category,
            importance=importance,
        )

    def _categorize(self, content: str) -> str:
        """Determine the most relevant memory category."""
        for category in _CATEGORY_PRIORITY:
            keywords = _CATEGORY_KEYWORDS[category]

            if any(keyword in content for keyword in keywords):
                return category

        return "general"

    def _calculate_importance(
        self,
        category: str,
        content: str,
    ) -> float:
        """Calculate the importance score of a memory."""
        importance_scores = {
            "project": 0.85,
            "goal": 0.85,
            "preference": 0.75,
            "education": 0.70,
            "work": 0.70,
            "personal": 0.65,
            "general": 0.50,
        }

        importance = importance_scores.get(category, 0.50)

        explicit_directives = (
            "remember that",
            "from now on",
            "going forward",
        )

        if any(
            directive in content
            for directive in explicit_directives
        ):
            importance += 0.10

        return min(importance, 1.0)

    def _is_duplicate(
        self,
        content: str,
        existing_memories: Sequence[Memory],
    ) -> bool:
        """Check whether an equivalent memory already exists."""
        normalized_content = self._normalize(content)

        for memory in existing_memories:
            normalized_existing = self._normalize(memory.content)

            if normalized_content == normalized_existing:
                return True

        return False

    def _split_sentences(self, content: str) -> list[str]:
        """Split message content into individual sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _normalize(self, content: str) -> str:
        """Normalize text for matching and duplicate detection."""
        normalized = content.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s'-]", "", normalized)

        return normalized
