
"""Tests for memory update execution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.conversations.memory_conflict_detector import (
    MemoryConflictType,
)
from seven.core.conversations.memory_update_policy import (
    MemoryUpdateAction,
)
from seven.core.conversations.memory_update_service import (
    MemoryUpdateService,
)

@pytest.fixture
def memory():
    return SimpleNamespace(
        id="memory-001",
        content="I prefer dark mode.",
        category="preference",
        importance=0.8,
    )

@pytest.fixture
def repository(memory):
    instance = SimpleNamespace(
        get_by_id=AsyncMock(return_value=memory),
        update=AsyncMock(
            side_effect=lambda entity, values: (
                setattr(entity, "content", values["content"]) or entity
            ),
        ),
    )
    return instance

@pytest.fixture
def service(repository):
    return MemoryUpdateService(repository)

@pytest.mark.asyncio
async def test_higher_confidence_update_changes_memory(
    service,
    repository,
    memory,
):
    result = await service.update_memory(
        memory.id,
        "I prefer light mode.",
        existing_confidence=0.6,
        incoming_confidence=0.9,
    )

    assert result is not None
    assert result.decision.action == MemoryUpdateAction.UPDATE
    assert result.was_updated is True
    assert memory.content == "I prefer light mode."
    repository.update.assert_awaited_once()

@pytest.mark.asyncio
async def test_lower_confidence_keeps_existing_memory(
    service,
    repository,
    memory,
):
    result = await service.update_memory(
        memory.id,
        "I prefer light mode.",
        existing_confidence=0.9,
        incoming_confidence=0.6,
    )

    assert result is not None
    assert result.decision.action == MemoryUpdateAction.KEEP
    assert result.was_updated is False
    assert memory.content == "I prefer dark mode."
    repository.update.assert_not_awaited()

@pytest.mark.asyncio
async def test_conflicting_similar_confidence_requires_review(
    service,
    repository,
    memory,
):
    result = await service.update_memory(
        memory.id,
        "I prefer light mode.",
        existing_confidence=0.8,
        incoming_confidence=0.82,
    )

    assert result is not None
    assert result.decision.action == MemoryUpdateAction.REVIEW
    assert result.conflict.conflict_type == (
        MemoryConflictType.DIRECT_CONTRADICTION
    )
    assert result.was_updated is False
    repository.update.assert_not_awaited()

@pytest.mark.asyncio
async def test_identical_memory_is_reinforced(
    service,
    repository,
    memory,
):
    result = await service.update_memory(
        memory.id,
        "I prefer dark mode.",
        existing_confidence=0.8,
        incoming_confidence=0.9,
    )

    assert result is not None
    assert result.decision.action == MemoryUpdateAction.REINFORCE
    assert result.was_updated is False
    repository.update.assert_not_awaited()

@pytest.mark.asyncio
async def test_missing_memory_returns_none(service, repository):
    repository.get_by_id.return_value = None

    result = await service.update_memory(
        "missing-memory",
        "New information",
        existing_confidence=0.5,
        incoming_confidence=0.9,
    )

    assert result is None
    repository.update.assert_not_awaited()

@pytest.mark.asyncio
async def test_update_uses_repository_memory_id(repository, service):
    await service.update_memory(
        "memory-001",
        "I prefer light mode.",
        existing_confidence=0.6,
        incoming_confidence=0.9,
    )

    repository.get_by_id.assert_awaited_once_with("memory-001")

@pytest.mark.asyncio
async def test_invalid_memory_id_is_rejected(service):
    with pytest.raises(ValueError):
        await service.update_memory(
            "",
            "New information",
            existing_confidence=0.5,
            incoming_confidence=0.9,
        )

@pytest.mark.asyncio
async def test_empty_incoming_content_is_rejected(service):
    with pytest.raises(ValueError):
        await service.update_memory(
            "memory-001",
            " ",
            existing_confidence=0.5,
            incoming_confidence=0.9,
        )