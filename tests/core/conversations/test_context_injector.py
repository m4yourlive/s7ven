
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.models import AIMessage, MessageRole
from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_service import MemoryService

@pytest.fixture
def memory_retriever():
    service = MemoryService(session=AsyncMock())
    service._repository.search = AsyncMock(return_value=[])

    return MemoryRetriever(service)


@pytest.fixture
def injector(memory_retriever):
    return ContextInjector(
        memory_retriever,
        max_memories=3,
    )


@pytest.fixture
def sample_memory():
    return SimpleNamespace(
        id="memory-1",
        content="User is building the S7VEN AI assistant.",
        category="project",
        importance=0.9,
        created_at=datetime.now(timezone.utc),
    )


def test_init_with_valid_retriever(memory_retriever):
    injector = ContextInjector(memory_retriever)

    assert injector._memory_retriever is memory_retriever
    assert injector._max_memories == 5


def test_init_rejects_invalid_retriever():
    with pytest.raises(TypeError):
        ContextInjector(object())


@pytest.mark.parametrize("max_memories", [0, -1, 1.5, True])
def test_init_rejects_invalid_max_memories(memory_retriever, max_memories):
    with pytest.raises(ValueError):
        ContextInjector(
            memory_retriever,
            max_memories=max_memories,
        )


@pytest.mark.asyncio
async def test_inject_rejects_empty_message(injector):
    with pytest.raises(ValueError, match="Message cannot be empty"):
        await injector.inject("   ")


@pytest.mark.asyncio
async def test_inject_rejects_invalid_history(injector):
    with pytest.raises(TypeError, match="history must be a tuple"):
        await injector.inject(
            "Hello",
            history=[],
        )


@pytest.mark.asyncio
async def test_inject_rejects_invalid_history_items(injector):
    with pytest.raises(TypeError, match="only AIMessage"):
        await injector.inject(
            "Hello",
            history=("invalid",),
        )


@pytest.mark.asyncio
async def test_inject_adds_relevant_memories(
    memory_retriever,
    sample_memory,
):
    memory_retriever._memory_service._repository.search.return_value = [
        sample_memory
    ]

    injector = ContextInjector(memory_retriever)

    result = await injector.inject("Tell me about S7VEN")

    assert len(result) == 1
    assert result[0].role == MessageRole.SYSTEM
    assert "S7VEN AI assistant" in result[0].content

    memory_retriever._memory_service._repository.search.assert_awaited_once_with(
        "Tell me about S7VEN",
        limit=25,
    )


@pytest.mark.asyncio
async def test_inject_preserves_existing_history(
    memory_retriever,
    sample_memory,
):
    memory_retriever._memory_service._repository.search.return_value = [
        sample_memory
    ]

    injector = ContextInjector(memory_retriever)

    history = (
        AIMessage(
            role=MessageRole.USER,
            content="Hello",
        ),
        AIMessage(
            role=MessageRole.ASSISTANT,
            content="Hi!",
        ),
    )

    result = await injector.inject(
        "Tell me about S7VEN",
        history,
    )

    assert len(result) == 3
    assert result[0].role == MessageRole.SYSTEM
    assert result[1:] == history


@pytest.mark.asyncio
async def test_inject_returns_history_when_no_memories(
    injector,
    memory_retriever,
):
    history = (
        AIMessage(
            role=MessageRole.USER,
            content="Hello",
        ),
    )

    result = await injector.inject("Hello", history)

    assert result == history
    memory_retriever._memory_service._repository.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_inject_respects_max_memories(
    memory_retriever,
):
    injector = ContextInjector(
        memory_retriever,
        max_memories=20,
    )

    await injector.inject("S7VEN")

    memory_retriever._memory_service._repository.search.assert_awaited_once_with(
        "S7VEN",
        limit=100,
    )
