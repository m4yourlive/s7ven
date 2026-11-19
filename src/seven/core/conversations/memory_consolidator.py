
"""Consolidate extracted memory candidates into persistent storage."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from seven.core.conversations.memory_conflict_detector import (
    MemoryConflict,
    MemoryConflictDetector,
)
from seven.core.conversations.memory_contracts import MemoryCandidate
from seven.core.conversations.memory_extractor import MemoryExtractor
from seven.core.conversations.memory_normalizer import (
    MemoryCandidateNormalizer,
)
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.memory_update_policy import (
    MemoryUpdateAction,
)
from seven.core.conversations.memory_update_service import (
    MemoryUpdateResult,
    MemoryUpdateService,
)
from seven.core.conversations.message_service import MessageService
from seven.core.database.models import Memory, Message


class MemoryConsolidationAction(StrEnum):
    """Actions produced by the memory consolidation engine."""

    CREATE = "create"
    REINFORCE = "reinforce"
    UPDATE = "update"
    KEEP = "keep"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class MemoryConsolidationItem:
    """Represent the outcome of consolidating one candidate."""

    candidate: MemoryCandidate
    action: MemoryConsolidationAction
    memory: Memory | None = None
    conflict: MemoryConflict | None = None
    update_result: MemoryUpdateResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MemoryCandidate):
            raise TypeError("Candidate must be a MemoryCandidate.")

        if not isinstance(self.action, MemoryConsolidationAction):
            raise TypeError(
                "Action must be a MemoryConsolidationAction."
            )

        if self.conflict is not None and not isinstance(
            self.conflict,
            MemoryConflict,
        ):
            raise TypeError("Conflict must be a MemoryConflict.")

        if self.update_result is not None and not isinstance(
            self.update_result,
            MemoryUpdateResult,
        ):
            raise TypeError(
                "Update result must be a MemoryUpdateResult."
            )


@dataclass(frozen=True, slots=True)
class MemoryConsolidationResult:
    """Represent the outcomes of a consolidation batch."""

    items: tuple[MemoryConsolidationItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("Items must be a tuple.")

        if any(
            not isinstance(item, MemoryConsolidationItem)
            for item in self.items
        ):
            raise TypeError(
                "All items must be MemoryConsolidationItem instances."
            )

    @property
    def created(self) -> int:
        return self._count(MemoryConsolidationAction.CREATE)

    @property
    def reinforced(self) -> int:
        return self._count(MemoryConsolidationAction.REINFORCE)

    @property
    def updated(self) -> int:
        return self._count(MemoryConsolidationAction.UPDATE)

    @property
    def kept(self) -> int:
        return self._count(MemoryConsolidationAction.KEEP)

    @property
    def needs_review(self) -> int:
        return self._count(MemoryConsolidationAction.REVIEW)

    def _count(self, action: MemoryConsolidationAction) -> int:
        return sum(item.action == action for item in self.items)


class MemoryConsolidator:
    """Coordinate memory extraction, consolidation, and persistence."""

    def __init__(
        self,
        message_service: MessageService,
        memory_service: MemoryService,
        extractor: MemoryExtractor | None = None,
        *,
        update_service: MemoryUpdateService | None = None,
        conflict_detector: MemoryConflictDetector | None = None,
        existing_confidence: float = 0.5,
        search_limit: int = 20,
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

        if update_service is not None and not isinstance(
            update_service,
            MemoryUpdateService,
        ):
            raise TypeError(
                "update_service must be a MemoryUpdateService instance."
            )

        if conflict_detector is not None and not isinstance(
            conflict_detector,
            MemoryConflictDetector,
        ):
            raise TypeError(
                "conflict_detector must be a MemoryConflictDetector instance."
            )

        if (
            isinstance(existing_confidence, bool)
            or not isinstance(existing_confidence, (int, float))
            or not 0.0 <= existing_confidence <= 1.0
        ):
            raise ValueError(
                "Existing confidence must be between 0.0 and 1.0."
            )

        if (
            isinstance(search_limit, bool)
            or not isinstance(search_limit, int)
            or search_limit < 1
        ):
            raise ValueError("Search limit must be greater than zero.")

        self.message_service = message_service
        self.memory_service = memory_service
        self.extractor = extractor or MemoryExtractor()

        self.conflict_detector = (
            conflict_detector or MemoryConflictDetector()
        )
        self.update_service = update_service or MemoryUpdateService(
            memory_service.repository,
            conflict_detector=self.conflict_detector,
        )

        self.existing_confidence = float(existing_confidence)
        self.search_limit = search_limit

    async def consolidate(
        self,
        conversation_id: str,
    ) -> list[Memory]:
        """Extract and persist newly created memories from a conversation.

        This method preserves the original return behavior: only newly
        created memories are returned.
        """
        result = await self.consolidate_detailed(conversation_id)

        return [
            item.memory
            for item in result.items
            if (
                item.action == MemoryConsolidationAction.CREATE
                and item.memory is not None
            )
        ]

    async def consolidate_detailed(
        self,
        conversation_id: str,
    ) -> MemoryConsolidationResult:
        """Extract candidates and return detailed consolidation outcomes."""
        if not isinstance(conversation_id, str):
            raise TypeError("Conversation ID must be a string.")

        normalized_id = conversation_id.strip()

        if not normalized_id:
            raise ValueError("Conversation ID cannot be empty.")

        messages = await self.message_service.get_messages(
            normalized_id
        )
        candidates = self.extract_candidates(messages)

        items: list[MemoryConsolidationItem] = []

        for candidate in candidates:
            item = await self._consolidate_candidate(
                candidate,
                conversation_id=normalized_id,
            )
            items.append(item)

        return MemoryConsolidationResult(items=tuple(items))

    async def _consolidate_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        conversation_id: str,
    ) -> MemoryConsolidationItem:
        """Evaluate and consolidate one candidate."""
        existing_memories = await self.memory_service.search(
            candidate.content,
            limit=self.search_limit,
        )

        exact_match = next(
            (
                memory
                for memory in existing_memories
                if MemoryCandidateNormalizer.are_duplicates(
                    candidate.content,
                    memory.content,
                )
            ),
            None,
        )

        if exact_match is not None:
            if not hasattr(exact_match, "id"):
                return MemoryConsolidationItem(
                    candidate=candidate,
                    action=MemoryConsolidationAction.KEEP,
                    memory=exact_match,
                )

            return await self._reinforce(
                candidate,
                exact_match,
            )

        conflicts: list[tuple[Memory, MemoryConflict]] = []

        for memory in existing_memories:
            conflict = self.conflict_detector.detect(
                memory.content,
                candidate.content,
            )

            if conflict.is_conflict:
                conflicts.append((memory, conflict))

        if conflicts:
            memory, conflict = conflicts[0]

            if conflict.conflict_type.value == "possible_conflict":
                return MemoryConsolidationItem(
                    candidate=candidate,
                    action=MemoryConsolidationAction.REVIEW,
                    memory=memory,
                    conflict=conflict,
                )

            return await self._resolve_conflict(
                candidate,
                memory,
                conflict,
            )

        memory = await self.memory_service.create(
            conversation_id=conversation_id,
            content=candidate.content,
            category=str(candidate.category),
            importance=candidate.importance,
        )

        return MemoryConsolidationItem(
            candidate=candidate,
            action=MemoryConsolidationAction.CREATE,
            memory=memory,
        )

    async def _reinforce(
        self,
        candidate: MemoryCandidate,
        memory: Memory,
    ) -> MemoryConsolidationItem:
        """Reinforce an exact duplicate without replacing its content."""
        update_result = await self.update_service.update_memory(
            memory.id,
            candidate.content,
            existing_confidence=self.existing_confidence,
            incoming_confidence=candidate.confidence,
        )

        if update_result is None:
            return MemoryConsolidationItem(
                candidate=candidate,
                action=MemoryConsolidationAction.REVIEW,
                memory=memory,
            )

        if (
            update_result.decision.action
            != MemoryUpdateAction.REINFORCE
        ):
            return self._build_update_item(
                candidate,
                update_result,
            )

        reinforced_importance = max(
            update_result.memory.importance,
            candidate.importance,
        )

        reinforced_memory = await self.memory_service.update(
            update_result.memory.id,
            importance=reinforced_importance,
        )

        return MemoryConsolidationItem(
            candidate=candidate,
            action=MemoryConsolidationAction.REINFORCE,
            memory=reinforced_memory or update_result.memory,
            conflict=update_result.conflict,
            update_result=update_result,
        )

    async def _resolve_conflict(
        self,
        candidate: MemoryCandidate,
        memory: Memory,
        conflict: MemoryConflict,
    ) -> MemoryConsolidationItem:
        """Resolve a detected conflict using the update decision policy."""
        update_result = await self.update_service.update_memory(
            memory.id,
            candidate.content,
            existing_confidence=self.existing_confidence,
            incoming_confidence=candidate.confidence,
        )

        if update_result is None:
            return MemoryConsolidationItem(
                candidate=candidate,
                action=MemoryConsolidationAction.REVIEW,
                memory=memory,
                conflict=conflict,
            )

        return self._build_update_item(
            candidate,
            update_result,
            conflict=conflict,
        )

    @staticmethod
    def _build_update_item(
        candidate: MemoryCandidate,
        update_result: MemoryUpdateResult,
        *,
        conflict: MemoryConflict | None = None,
    ) -> MemoryConsolidationItem:
        """Translate an update decision into a consolidation outcome."""
        decision_action = update_result.decision.action

        if (
            decision_action == MemoryUpdateAction.UPDATE
            and update_result.was_updated
        ):
            action = MemoryConsolidationAction.UPDATE
        elif decision_action == MemoryUpdateAction.KEEP:
            action = MemoryConsolidationAction.KEEP
        elif decision_action == MemoryUpdateAction.REINFORCE:
            action = MemoryConsolidationAction.REINFORCE
        else:
            action = MemoryConsolidationAction.REVIEW

        return MemoryConsolidationItem(
            candidate=candidate,
            action=action,
            memory=update_result.memory,
            conflict=conflict or update_result.conflict,
            update_result=update_result,
        )

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
