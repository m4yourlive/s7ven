"""Consolidate extracted memory candidates into persistent storage."""

from __future__ import annotations

import re
from collections.abc import Sequence

from seven.core.conversations.memory_contracts import MemoryCandidate
from seven.core.conversations.memory_extractor import MemoryExtractor
from seven.core.conversations.memory_normalizer import (
    MemoryCandidateNormalizer,
)
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.message_service import MessageService
from seven.core.database.models import Memory, Message

class MemoryConsolidator:
    """Coordinate memory extraction, deduplication, and persistence."""

    def __init__(
        self,
        message_service: MessageService,
        memory_service: MemoryService,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        """Initialize the memory consolidator."""
        if not isinstance(message_service, MessageService):
            raise TypeError(
                "message_service must be a MessageService instance."
            )

        if not isinstance(memory_service, MemoryService):
            raise TypeError(
                "memory_service must be a MemoryService instance."
            )

        if extractor is not None and not isinstance(
            extractor,
            MemoryExtractor,
        ):
            raise TypeError(
                "extractor must be a MemoryExtractor instance."
            )

        self.message_service = message_service
        self.memory_service = memory_service
        self.extractor = extractor or MemoryExtractor()

    async def consolidate(
        self,
        conversation_id: str,
    ) -> list[Memory]:
        """Extract and persist new memories from a conversation."""
        if not isinstance(conversation_id, str):
            raise TypeError("Conversation ID must be a string.")

        normalized_id = conversation_id.strip()

        if not normalized_id:
            raise ValueError("Conversation ID cannot be empty.")

        messages = await self.message_service.get_messages(
            normalized_id
        )
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
                conversation_id=normalized_id,
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
        """Extract and deduplicate candidates from user messages."""
        candidates: list[MemoryCandidate] = []

        for message in messages:
            if message.role != "user":
                continue

            for sentence in self._split_sentences(message.content):
                candidate = self._build_candidate(sentence)

                if candidate is not None:
                    candidates.append(candidate)

        return MemoryCandidateNormalizer.deduplicate_candidates(
            candidates
        )

    def _build_candidate(
        self,
        sentence: str,
    ) -> MemoryCandidate | None:
        """Build a candidate through the shared memory extractor."""
        if not isinstance(sentence, str):
            raise TypeError("Candidate sentence must be a string.")

        extracted = self.extractor.extract(sentence)

        if not extracted:
            return None

        candidate = extracted[0]

        return MemoryCandidate(
            content=sentence.strip(),
            category=candidate.category,
            importance=candidate.importance,
            confidence=candidate.confidence,
            classification_confidence=(
                candidate.classification_confidence
            ),
            classification_evidence=candidate.classification_evidence,
        )

    def _categorize(self, content: str) -> str:
        """Determine the category using the shared extractor."""
        return self.extractor._classify_category(content)

    def _calculate_importance(
        self,
        category: str,
        content: str,
    ) -> float:
        """Calculate importance using the shared extractor."""
        return self.extractor._calculate_importance(
            content,
            category,
        )

    def _is_duplicate(
        self,
        content: str,
        existing_memories: Sequence[Memory],
    ) -> bool:
        """Check whether an equivalent memory already exists."""
        return any(
            MemoryCandidateNormalizer.are_duplicates(
                content,
                memory.content,
            )
            for memory in existing_memories
        )

    @staticmethod
    def _split_sentences(content: str) -> list[str]:
        """Split message content into individual sentences."""
        sentences = re.split(r"(?<=[.!?])\s+|\n+", content.strip())

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _normalize(content: str) -> str:
        """Normalize text for matching and duplicate detection."""
        return MemoryCandidateNormalizer.normalize_text(content)