
"""Tests for adaptive memory retrieval."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.conversations import (
    MemoryArchitectureConfig,
    MemoryRetriever,
    MemoryService,
)


@pytest.fixture
def memory_service():
    """Create a memory service with a mocked search operation."""
    service = MemoryService(session=AsyncMock())
    service.search = AsyncMock(return_value=[])
    return service


def make_memory(
    memory_id: str,
    content: str,
    importance: float = 0.8,
):
    """Create a lightweight memory record."""
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category="personal",
        importance=importance,
        conversation_id=None,
        created_at=None,
    )


@pytest.mark.asyncio
async def test_adaptive_retrieval_expands_when_results_are_insufficient(
    memory_service,
):
    irrelevant_memories = [
        make_memory(
            f"irrelevant-{index}",
            "I enjoy cooking at home.",
        )
        for index in range(20)
    ]

    relevant_memory = make_memory(
        "relevant",
        "I am a software developer.",
    )

    async def search(query: str, *, limit: int):
        if limit == 20:
            return irrelevant_memories

        if limit == 40:
            return [*irrelevant_memories, relevant_memory]

        return []

    memory_service.search = AsyncMock(side_effect=search)

    retriever = MemoryRetriever(memory_service)

    results = await retriever.retrieve(
        "software developer",
        limit=1,
    )

    assert [memory.id for memory in results] == ["relevant"]

    assert [
        call.kwargs["limit"]
        for call in memory_service.search.await_args_list
    ] == [20, 40]


@pytest.mark.asyncio
async def test_adaptive_retrieval_stops_when_enough_results_are_found(
    memory_service,
):
    memories = [
        make_memory(
            "relevant",
            "I am a software developer.",
        ),
        *[
            make_memory(
                f"irrelevant-{index}",
                "I enjoy cooking at home.",
            )
            for index in range(19)
        ],
    ]

    memory_service.search.return_value = memories

    retriever = MemoryRetriever(memory_service)

    results = await retriever.retrieve(
        "software developer",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].id == "relevant"

    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=20,
    )


@pytest.mark.asyncio
async def test_adaptive_retrieval_stops_when_candidates_are_exhausted(
    memory_service,
):
    memory_service.search.return_value = [
        make_memory(
            "irrelevant-1",
            "I enjoy cooking.",
        ),
        make_memory(
            "irrelevant-2",
            "I enjoy reading.",
        ),
    ]

    retriever = MemoryRetriever(memory_service)

    results = await retriever.retrieve("software developer")

    assert results == []

    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=25,
    )


@pytest.mark.asyncio
async def test_adaptive_retrieval_respects_maximum_candidate_limit(
    memory_service,
):
    config = MemoryArchitectureConfig(
        adaptive_retrieval_max_candidates=30,
    )

    irrelevant_memories = [
        make_memory(
            f"irrelevant-{index}",
            "I enjoy cooking at home.",
        )
        for index in range(30)
    ]

    memory_service.search.return_value = irrelevant_memories

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    results = await retriever.retrieve(
        "software developer",
        limit=1,
    )

    assert results == []

    assert [
        call.kwargs["limit"]
        for call in memory_service.search.await_args_list
    ] == [20, 30]


@pytest.mark.asyncio
async def test_adaptive_retrieval_respects_expansion_factor(
    memory_service,
):
    config = MemoryArchitectureConfig(
        adaptive_retrieval_expansion_factor=3,
        adaptive_retrieval_max_candidates=60,
    )

    irrelevant_memories = [
        make_memory(
            f"irrelevant-{index}",
            "I enjoy cooking at home.",
        )
        for index in range(60)
    ]

    memory_service.search.return_value = irrelevant_memories

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    await retriever.retrieve(
        "software developer",
        limit=1,
    )

    assert [
        call.kwargs["limit"]
        for call in memory_service.search.await_args_list
    ] == [20, 60]


@pytest.mark.asyncio
async def test_adaptive_retrieval_can_be_disabled(
    memory_service,
):
    config = MemoryArchitectureConfig(
        adaptive_retrieval_enabled=False,
    )

    memory_service.search.return_value = [
        make_memory(
            "irrelevant",
            "I enjoy cooking at home.",
        )
    ]

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    results = await retriever.retrieve("software developer")

    assert results == []

    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=25,
    )


def test_adaptive_retrieval_rejects_invalid_expansion_factor():
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            adaptive_retrieval_expansion_factor=1,
        )


def test_adaptive_retrieval_rejects_maximum_below_minimum_candidates():
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            retrieval_min_candidates=30,
            adaptive_retrieval_max_candidates=20,
        )


def test_adaptive_retrieval_rejects_non_boolean_enabled_value():
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(
            adaptive_retrieval_enabled=1,
        )
