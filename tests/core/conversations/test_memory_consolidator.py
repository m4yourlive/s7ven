
"""Tests for memory consolidation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.conversations.memory_conflict_detector import (
    MemoryConflict,
    MemoryConflictType,
)
from seven.core.conversations.memory_consolidator import (
    MemoryConsolidationAction,
    MemoryConsolidationItem,
    MemoryConsolidationResult,
    MemoryConsolidator,
)
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.message_service import MessageService


@pytest.fixture
def services():
    """Create mocked conversation and memory services."""
    message_service = AsyncMock(spec=MessageService)
    memory_service = AsyncMock(spec=MemoryService)

    return message_service, memory_service


@pytest.fixture
def consolidator(services):
    """Create a memory consolidator with mocked services."""
    message_service, memory_service = services

    return MemoryConsolidator(
        message_service=message_service,
        memory_service=memory_service,
    )


@pytest.fixture
def candidate(consolidator):
    """Create a representative memory candidate."""
    return consolidator.extract_candidates(
        [
            SimpleNamespace(
                role="user",
                content=(
                    "I am building a personal AI assistant "
                    "called S7VEN."
                ),
            ),
        ]
    )[0]


def test_extracts_user_memory_candidates(consolidator):
    messages = [
        SimpleNamespace(
            role="user",
            content="I am building a personal AI assistant called S7VEN.",
        ),
        SimpleNamespace(
            role="assistant",
            content="That sounds interesting.",
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert len(candidates) == 1
    assert candidates[0].category == "project"
    assert "S7VEN" in candidates[0].content


def test_ignores_non_memory_messages(consolidator):
    messages = [
        SimpleNamespace(
            role="user",
            content="What is the weather today?",
        ),
        SimpleNamespace(
            role="assistant",
            content="I am happy to help.",
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert candidates == []


def test_categorizes_preferences(consolidator):
    messages = [
        SimpleNamespace(
            role="user",
            content="I prefer minimal and clean interface designs.",
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert len(candidates) == 1
    assert candidates[0].category == "preference"
    assert candidates[0].importance == 0.75


def test_categorizes_goals(consolidator):
    messages = [
        SimpleNamespace(
            role="user",
            content="I want to become a professional software developer.",
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert len(candidates) == 1
    assert candidates[0].category == "goal"
    assert candidates[0].importance == 0.85


def test_applies_importance_bonus_for_explicit_memory_instruction(
    consolidator,
):
    messages = [
        SimpleNamespace(
            role="user",
            content="From now on, I prefer concise technical explanations.",
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert len(candidates) == 1
    assert candidates[0].importance == 0.85


def test_removes_duplicate_candidates(consolidator):
    messages = [
        SimpleNamespace(
            role="user",
            content=(
                "I am building a personal AI assistant. "
                "I am building a personal AI assistant."
            ),
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert len(candidates) == 1


def test_ignores_empty_and_oversized_messages(consolidator):
    messages = [
        SimpleNamespace(role="user", content=""),
        SimpleNamespace(
            role="user",
            content="I am " + ("building " * 100),
        ),
    ]

    candidates = consolidator.extract_candidates(messages)

    assert candidates == []


@pytest.mark.asyncio
async def test_consolidate_persists_new_memories(
    consolidator,
    services,
):
    message_service, memory_service = services

    message_service.get_messages.return_value = [
        SimpleNamespace(
            role="user",
            content="I am building a personal AI assistant called S7VEN.",
        ),
    ]

    memory_service.search.return_value = []
    memory_service.create.return_value = SimpleNamespace(
        id="memory-1",
        content="I am building a personal AI assistant called S7VEN.",
        category="project",
        importance=0.85,
    )

    result = await consolidator.consolidate("conversation-1")

    assert len(result) == 1

    memory_service.create.assert_awaited_once_with(
        content="I am building a personal AI assistant called S7VEN.",
        category="project",
        importance=0.85,
        conversation_id="conversation-1",
    )


@pytest.mark.asyncio
async def test_consolidate_skips_existing_memories(
    consolidator,
    services,
):
    message_service, memory_service = services

    content = "I am building a personal AI assistant called S7VEN."

    message_service.get_messages.return_value = [
        SimpleNamespace(role="user", content=content),
    ]

    memory_service.search.return_value = [
        SimpleNamespace(content=content),
    ]

    result = await consolidator.consolidate("conversation-1")

    assert result == []
    memory_service.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_consolidate_skips_duplicate_candidates(
    consolidator,
    services,
):
    message_service, memory_service = services

    content = "I am building a personal AI assistant called S7VEN."

    message_service.get_messages.return_value = [
        SimpleNamespace(
            role="user",
            content=f"{content} {content}",
        ),
    ]

    memory_service.search.return_value = []
    memory_service.create.return_value = SimpleNamespace(
        id="memory-1",
        content=content,
        category="project",
        importance=0.85,
    )

    result = await consolidator.consolidate("conversation-1")

    assert len(result) == 1
    memory_service.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_consolidate_returns_empty_when_no_candidates(
    consolidator,
    services,
):
    message_service, memory_service = services

    message_service.get_messages.return_value = [
        SimpleNamespace(
            role="user",
            content="Can you explain how databases work?",
        ),
    ]

    result = await consolidator.consolidate("conversation-1")

    assert result == []
    memory_service.search.assert_not_awaited()
    memory_service.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_detailed_consolidation_reports_created_memories(
    consolidator,
    services,
):
    message_service, memory_service = services

    message_service.get_messages.return_value = [
        SimpleNamespace(
            role="user",
            content="I am building a personal AI assistant called S7VEN.",
        ),
    ]

    created_memory = SimpleNamespace(
        id="memory-1",
        content="I am building a personal AI assistant called S7VEN.",
        category="project",
        importance=0.85,
    )

    memory_service.search.return_value = []
    memory_service.create.return_value = created_memory

    result = await consolidator.consolidate_detailed("conversation-1")

    assert isinstance(result, MemoryConsolidationResult)
    assert len(result.items) == 1
    assert result.items[0].action == MemoryConsolidationAction.CREATE
    assert result.items[0].memory is created_memory
    assert result.created == 1
    assert result.reinforced == 0
    assert result.updated == 0
    assert result.kept == 0
    assert result.needs_review == 0


@pytest.mark.asyncio
async def test_detailed_consolidation_reports_existing_memory_as_kept(
    consolidator,
    services,
):
    message_service, memory_service = services

    content = "I am building a personal AI assistant called S7VEN."
    existing_memory = SimpleNamespace(content=content)

    message_service.get_messages.return_value = [
        SimpleNamespace(role="user", content=content),
    ]
    memory_service.search.return_value = [existing_memory]

    result = await consolidator.consolidate_detailed("conversation-1")

    assert len(result.items) == 1
    assert result.items[0].action == MemoryConsolidationAction.KEEP
    assert result.items[0].memory is existing_memory
    assert result.kept == 1
    memory_service.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_detailed_consolidation_routes_exact_matches_to_reinforcement(
    consolidator,
    services,
    monkeypatch,
    candidate,
):
    message_service, memory_service = services

    existing_memory = SimpleNamespace(
        id="memory-1",
        content=candidate.content,
    )

    message_service.get_messages.return_value = [
        SimpleNamespace(role="user", content=candidate.content),
    ]
    memory_service.search.return_value = [existing_memory]

    reinforced_item = MemoryConsolidationItem(
        candidate=candidate,
        action=MemoryConsolidationAction.REINFORCE,
        memory=existing_memory,
    )

    reinforce = AsyncMock(return_value=reinforced_item)
    monkeypatch.setattr(consolidator, "_reinforce", reinforce)

    result = await consolidator.consolidate_detailed("conversation-1")

    assert len(result.items) == 1
    assert result.items[0].action == MemoryConsolidationAction.REINFORCE
    assert result.reinforced == 1

    reinforce.assert_awaited_once_with(
        candidate,
        existing_memory,
    )


@pytest.mark.asyncio
async def test_detailed_consolidation_marks_possible_conflicts_for_review(
    consolidator,
    services,
    monkeypatch,
    candidate,
):
    message_service, memory_service = services

    existing_memory = SimpleNamespace(
        id="memory-2",
        content="I prefer a different personal assistant project.",
    )

    conflict = MemoryConflict(
        conflict_type=MemoryConflictType.POSSIBLE_CONFLICT,
        is_conflict=True,
        shared_terms=("personal", "assistant"),
        reason="The memories may describe conflicting information.",
    )

    message_service.get_messages.return_value = [
        SimpleNamespace(role="user", content=candidate.content),
    ]
    memory_service.search.return_value = [existing_memory]

    monkeypatch.setattr(
        consolidator.conflict_detector,
        "detect",
        lambda existing, incoming: conflict,
    )

    result = await consolidator.consolidate_detailed("conversation-1")

    assert len(result.items) == 1
    assert result.items[0].action == MemoryConsolidationAction.REVIEW
    assert result.items[0].memory is existing_memory
    assert result.items[0].conflict is conflict
    assert result.needs_review == 1

    memory_service.create.assert_not_awaited()


def test_consolidation_result_counts_all_actions(candidate):
    actions = (
        MemoryConsolidationAction.CREATE,
        MemoryConsolidationAction.REINFORCE,
        MemoryConsolidationAction.UPDATE,
        MemoryConsolidationAction.KEEP,
        MemoryConsolidationAction.REVIEW,
    )

    result = MemoryConsolidationResult(
        items=tuple(
            MemoryConsolidationItem(
                candidate=candidate,
                action=action,
            )
            for action in actions
        )
    )

    assert result.created == 1
    assert result.reinforced == 1
    assert result.updated == 1
    assert result.kept == 1
    assert result.needs_review == 1


@pytest.mark.asyncio
async def test_detailed_consolidation_rejects_empty_conversation_id(
    consolidator,
    services,
):
    message_service, _ = services

    with pytest.raises(ValueError, match="cannot be empty"):
        await consolidator.consolidate_detailed("   ")

    message_service.get_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_detailed_consolidation_rejects_non_string_conversation_id(
    consolidator,
    services,
):
    message_service, _ = services

    with pytest.raises(TypeError, match="must be a string"):
        await consolidator.consolidate_detailed(123)

    message_service.get_messages.assert_not_awaited()
