
"""Tests for memory consolidation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.conversations.memory_consolidator import (
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
