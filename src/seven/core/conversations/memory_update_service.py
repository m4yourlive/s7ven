
"""Execute memory updates using conflict detection and update policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seven.core.conversations.memory_conflict_detector import (
    MemoryConflict,
    MemoryConflictDetector,
)
from seven.core.conversations.memory_update_policy import (
    MemoryUpdateDecision,
    MemoryUpdatePolicy,
)

if TYPE_CHECKING:
    from seven.core.database.models.memory import Memory
    from seven.core.database.repositories.memory_repository import (
        MemoryRepository,
    )


@dataclass(frozen=True, slots=True)
class MemoryUpdateResult:
    """Represent the result of a memory update operation."""

    memory: Memory
    decision: MemoryUpdateDecision
    conflict: MemoryConflict
    was_updated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.decision, MemoryUpdateDecision):
            raise TypeError(
                "Decision must be a MemoryUpdateDecision."
            )

        if not isinstance(self.conflict, MemoryConflict):
            raise TypeError(
                "Conflict must be a MemoryConflict."
            )

        if not isinstance(self.was_updated, bool):
            raise TypeError("was_updated must be a boolean.")


class MemoryUpdateService:
    """Coordinate memory conflict detection and update execution."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        policy: MemoryUpdatePolicy | None = None,
        conflict_detector: MemoryConflictDetector | None = None,
    ) -> None:
        """Initialize the memory update service."""
        if repository is None:
            raise ValueError("Memory repository is required.")

        if policy is not None and not isinstance(
            policy,
            MemoryUpdatePolicy,
        ):
            raise TypeError(
                "Policy must be a MemoryUpdatePolicy."
            )

        if conflict_detector is not None and not isinstance(
            conflict_detector,
            MemoryConflictDetector,
        ):
            raise TypeError(
                "Conflict detector must be a MemoryConflictDetector."
            )

        self._repository = repository
        self._policy = policy or MemoryUpdatePolicy()
        self._conflict_detector = (
            conflict_detector or MemoryConflictDetector()
        )

    @property
    def repository(self) -> MemoryRepository:
        return self._repository

    @property
    def policy(self) -> MemoryUpdatePolicy:
        return self._policy

    @property
    def conflict_detector(self) -> MemoryConflictDetector:
        return self._conflict_detector

    async def update_memory(
        self,
        memory_id: str,
        incoming_content: str,
        *,
        existing_confidence: float,
        incoming_confidence: float,
    ) -> MemoryUpdateResult | None:
        """Evaluate and conditionally update an existing memory."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("Memory ID must be a non-empty string.")

        if (
            not isinstance(incoming_content, str)
            or not incoming_content.strip()
        ):
            raise ValueError("Incoming content must not be empty.")

        memory = await self._repository.get_by_id(memory_id)

        if memory is None:
            return None

        existing_content = memory.content

        conflict = self._conflict_detector.detect(
            existing_content,
            incoming_content,
        )

        decision = self._policy.evaluate(
            existing_content=existing_content,
            incoming_content=incoming_content,
            existing_confidence=existing_confidence,
            incoming_confidence=incoming_confidence,
        )

        was_updated = False

        if decision.action.value == "update":
            memory = await self._repository.update(
                memory,
                {"content": incoming_content.strip()},
            )
            was_updated = True

        return MemoryUpdateResult(
            memory=memory,
            decision=decision,
            conflict=conflict,
            was_updated=was_updated,
        )
