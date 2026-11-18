"""Tests for memory retrieval and scoring integration."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.conversations import (
    MemoryArchitectureConfig,
    MemoryRetriever,
    MemoryScorer,
    MemoryScoringContext,
    MemoryService,
)


@pytest.fixture
def memory_service():
    """Create a memory service with a mocked search operation."""
    service = MemoryService(session=AsyncMock())
    service.search = AsyncMock(return_value=[])
    return service


@pytest.fixture
def retriever(memory_service):
    return MemoryRetriever(memory_service)


def make_memory(
    memory_id="memory-001",
    content="I am a software developer.",
    category="personal",
    importance=0.8,
    conversation_id="conversation-001",
    created_at=None,
):
    """Create a lightweight memory record."""
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category=category,
        importance=importance,
        conversation_id=conversation_id,
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_retrieve_returns_ranked_memories(
    memory_service,
    retriever,
):
    memories = [
        make_memory(
            memory_id="low",
            content="I enjoy cooking at home.",
            importance=0.5,
        ),
        make_memory(
            memory_id="high",
            content="I am a software developer.",
            importance=0.8,
        ),
    ]

    memory_service.search.return_value = memories

    results = await retriever.retrieve("software developer")

    assert len(results) == 1
    assert results[0].id == "high"


@pytest.mark.asyncio
async def test_retrieve_uses_configured_candidate_limit(
    memory_service,
    retriever,
):
    await retriever.retrieve("software developer")

    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=25,
    )


@pytest.mark.asyncio
async def test_retrieve_scales_candidate_limit(
    memory_service,
    retriever,
):
    await retriever.retrieve(
        "software developer",
        limit=10,
    )

    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=50,
    )


@pytest.mark.asyncio
async def test_retrieve_respects_custom_candidate_configuration(
    memory_service,
):
    config = MemoryArchitectureConfig(
        retrieval_candidate_multiplier=4,
        retrieval_min_candidates=30,
    )

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    await retriever.retrieve(
        "software developer",
        limit=10,
    )

    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=40,
    )


@pytest.mark.asyncio
async def test_retrieve_uses_default_result_limit(
    memory_service,
):
    retriever = MemoryRetriever(
        memory_service,
        max_results=3,
    )

    memories = [
        make_memory(
            memory_id=f"memory-{index}",
            content="I am a software developer.",
        )
        for index in range(5)
    ]

    memory_service.search.return_value = memories

    results = await retriever.retrieve("software developer")

    assert len(results) == 3


@pytest.mark.asyncio
async def test_retrieve_respects_importance_threshold(
    memory_service,
):
    retriever = MemoryRetriever(
        memory_service,
        min_importance=0.7,
    )

    memory_service.search.return_value = [
        make_memory(
            memory_id="low",
            importance=0.4,
        ),
        make_memory(
            memory_id="high",
            importance=0.9,
        ),
    ]

    results = await retriever.retrieve("software developer")

    assert len(results) == 1
    assert results[0].id == "high"


@pytest.mark.asyncio
async def test_retrieve_uses_configured_minimum_importance(
    memory_service,
):
    config = MemoryArchitectureConfig(
        retrieval_min_importance=0.75,
    )

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    memory_service.search.return_value = [
        make_memory(
            memory_id="low",
            importance=0.7,
        ),
        make_memory(
            memory_id="high",
            importance=0.8,
        ),
    ]

    results = await retriever.retrieve("software developer")

    assert [memory.id for memory in results] == ["high"]


@pytest.mark.asyncio
async def test_retrieve_uses_injected_scorer(memory_service):
    scorer = MemoryScorer(
        MemoryArchitectureConfig(
            relevance_weight=0.5,
            importance_weight=0.5,
        )
    )

    retriever = MemoryRetriever(
        memory_service,
        scorer=scorer,
    )

    memory_service.search.return_value = [
        make_memory(),
    ]

    results = await retriever.retrieve("software developer")

    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_uses_contextual_scoring(
    memory_service,
    retriever,
):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    memory_service.search.return_value = [
        make_memory(
            memory_id="personal",
            category="personal",
            created_at=reference_time,
        ),
        make_memory(
            memory_id="project",
            category="project",
            created_at=reference_time,
        ),
    ]

    context = MemoryScoringContext(
        preferred_categories=("project",),
        reference_time=reference_time,
    )

    results = await retriever.retrieve(
        "software developer",
        context=context,
    )

    assert len(results) == 2
    assert results[0].id == "project"


@pytest.mark.asyncio
async def test_retrieve_by_tokens_uses_scoring(
    memory_service,
    retriever,
):
    memory = make_memory()

    memory_service.search.return_value = [memory]

    results = await retriever.retrieve_by_tokens(
        ["software", "developer"]
    )

    assert len(results) == 1
    memory_service.search.assert_awaited_once_with(
        "software developer",
        limit=25,
    )


@pytest.mark.asyncio
async def test_retrieve_by_tokens_supports_context(
    memory_service,
    retriever,
):
    reference_time = datetime.now(timezone.utc)

    memory_service.search.return_value = [
        make_memory(
            category="project",
            created_at=reference_time,
        )
    ]

    context = MemoryScoringContext(
        preferred_categories=("project",),
        reference_time=reference_time,
    )

    results = await retriever.retrieve_by_tokens(
        ["software", "developer"],
        context=context,
    )

    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_no_candidates(
    retriever,
):
    assert await retriever.retrieve("software developer") == []


@pytest.mark.asyncio
async def test_retrieve_returns_empty_for_tokenless_query(
    memory_service,
    retriever,
):
    results = await retriever.retrieve("!!!")

    assert results == []
    memory_service.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_query(retriever):
    with pytest.raises(ValueError):
        await retriever.retrieve("   ")


@pytest.mark.asyncio
async def test_retrieve_rejects_non_string_query(retriever):
    with pytest.raises(ValueError):
        await retriever.retrieve(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
async def test_retrieve_rejects_invalid_limit(retriever, limit):
    with pytest.raises(ValueError):
        await retriever.retrieve(
            "software developer",
            limit=limit,
        )


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_context(retriever):
    with pytest.raises(TypeError):
        await retriever.retrieve(
            "software developer",
            context="invalid",
        )


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_minimum_importance(
    memory_service,
):
    with pytest.raises(ValueError):
        MemoryRetriever(
            memory_service,
            min_importance=1.5,
        )


@pytest.mark.asyncio
async def test_retrieve_rejects_non_numeric_minimum_importance(
    memory_service,
):
    with pytest.raises(ValueError):
        MemoryRetriever(
            memory_service,
            min_importance="0.5",
        )


@pytest.mark.asyncio
async def test_retrieve_rejects_nan_minimum_importance(
    memory_service,
):
    with pytest.raises(ValueError):
        MemoryRetriever(
            memory_service,
            min_importance=float("nan"),
        )


def test_retriever_rejects_invalid_scorer(memory_service):
    with pytest.raises(TypeError):
        MemoryRetriever(
            memory_service,
            scorer="invalid",
        )


def test_retriever_rejects_scorer_and_config_together(
    memory_service,
):
    with pytest.raises(ValueError):
        MemoryRetriever(
            memory_service,
            scorer=MemoryScorer(),
            config=MemoryArchitectureConfig(),
        )


@pytest.mark.asyncio
async def test_retrieve_by_tokens_rejects_string(retriever):
    with pytest.raises(TypeError):
        await retriever.retrieve_by_tokens("software")


@pytest.mark.asyncio
async def test_retrieve_by_tokens_rejects_empty_tokens(retriever):
    with pytest.raises(ValueError):
        await retriever.retrieve_by_tokens([])


@pytest.mark.asyncio
async def test_retrieve_by_tokens_rejects_invalid_tokens(retriever):
    with pytest.raises(ValueError):
        await retriever.retrieve_by_tokens(
            [" ", None, 123]
        )


def test_legacy_tokenize_remains_available():
    assert MemoryRetriever._tokenize(
        "Software Developer"
    ) == {"software", "developer"}


def test_legacy_relevance_score_remains_available():
    score = MemoryRetriever._relevance_score(
        {"software", "developer"},
        "I am a software developer.",
    )

    assert score == 1.0