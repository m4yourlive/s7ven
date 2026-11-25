"""Tests for memory scoring calibration and retrieval thresholds."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.conversations import (
    MemoryArchitectureConfig,
    MemoryScorer,
    MemoryService,
    MemoryRetriever,
)


@pytest.fixture
def memory_service():
    """Create a memory service with a mocked search operation."""
    service = MemoryService(session=AsyncMock())
    service.search = AsyncMock(return_value=[])
    return service


def make_memory(
    memory_id,
    content,
    importance=0.2,
    category="general",
):
    """Create a lightweight memory record."""
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category=category,
        importance=importance,
        conversation_id="conversation-001",
        created_at=None,
    )


def test_calibration_defaults_preserve_existing_weights():
    config = MemoryArchitectureConfig()

    assert config.lexical_coverage_weight == 0.55
    assert config.lexical_precision_weight == 0.15
    assert config.lexical_jaccard_weight == 0.15
    assert config.lexical_phrase_weight == 0.15

    assert config.retrieval_min_importance == 0.0
    assert config.retrieval_min_relevance == 0.0
    assert config.retrieval_min_combined_score == 0.0


def test_custom_lexical_weights_are_applied():
    config = MemoryArchitectureConfig(
        lexical_coverage_weight=0.4,
        lexical_precision_weight=0.1,
        lexical_jaccard_weight=0.1,
        lexical_phrase_weight=0.4,
    )

    scorer = MemoryScorer(config)

    result = scorer.calculate_relevance(
        "software developer",
        "I am software developer and Python engineer.",
    )

    expected = (
        result.token_coverage * 0.4
        + result.token_precision * 0.1
        + result.jaccard_similarity * 0.1
        + result.phrase_match * 0.4
    )

    assert result.relevance_score == pytest.approx(expected)


def test_custom_lexical_weights_change_calibration():
    default_scorer = MemoryScorer()

    custom_scorer = MemoryScorer(
        MemoryArchitectureConfig(
            lexical_coverage_weight=0.4,
            lexical_precision_weight=0.1,
            lexical_jaccard_weight=0.1,
            lexical_phrase_weight=0.4,
        )
    )

    query = "software developer"
    content = "I am software developer and Python engineer."

    default_score = default_scorer.calculate_relevance(
        query,
        content,
    ).relevance_score

    custom_score = custom_scorer.calculate_relevance(
        query,
        content,
    ).relevance_score

    assert custom_score > default_score


@pytest.mark.parametrize(
    "field",
    [
        "lexical_coverage_weight",
        "lexical_precision_weight",
        "lexical_jaccard_weight",
        "lexical_phrase_weight",
    ],
)
def test_config_rejects_invalid_lexical_weight_values(field):
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            **{field: 1.5}
        )


@pytest.mark.parametrize(
    "field",
    [
        "lexical_coverage_weight",
        "lexical_precision_weight",
        "lexical_jaccard_weight",
        "lexical_phrase_weight",
    ],
)
def test_config_rejects_non_numeric_lexical_weights(field):
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(
            **{field: "0.25"}
        )


def test_config_requires_lexical_weights_to_sum_to_one():
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            lexical_coverage_weight=0.5,
            lexical_precision_weight=0.1,
            lexical_jaccard_weight=0.1,
            lexical_phrase_weight=0.1,
        )


def test_config_accepts_custom_lexical_weights():
    config = MemoryArchitectureConfig(
        lexical_coverage_weight=0.4,
        lexical_precision_weight=0.2,
        lexical_jaccard_weight=0.2,
        lexical_phrase_weight=0.2,
    )

    assert config.lexical_coverage_weight == 0.4
    assert config.lexical_precision_weight == 0.2
    assert config.lexical_jaccard_weight == 0.2
    assert config.lexical_phrase_weight == 0.2


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_min_relevance",
        "retrieval_min_combined_score",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_config_rejects_invalid_retrieval_thresholds(
    field,
    value,
):
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            **{field: value}
        )


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_min_relevance",
        "retrieval_min_combined_score",
    ],
)
def test_config_rejects_non_numeric_retrieval_thresholds(field):
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(
            **{field: "0.5"}
        )


@pytest.mark.asyncio
async def test_retriever_filters_by_minimum_relevance(
    memory_service,
):
    config = MemoryArchitectureConfig(
        retrieval_min_relevance=0.7,
    )

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    exact_memory = make_memory(
        "exact",
        "I am a software developer.",
    )

    partial_memory = make_memory(
        "partial",
        "I build software products.",
    )

    memory_service.search.return_value = [
        exact_memory,
        partial_memory,
    ]

    results = await retriever.retrieve(
        "software developer"
    )

    assert [memory.id for memory in results] == ["exact"]


@pytest.mark.asyncio
async def test_retriever_filters_by_minimum_combined_score(
    memory_service,
):
    config = MemoryArchitectureConfig(
        retrieval_min_combined_score=0.5,
    )

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    strong_memory = make_memory(
        "strong",
        "I am a software developer.",
        importance=0.2,
    )

    weak_memory = make_memory(
        "weak",
        "I build software products.",
        importance=0.2,
    )

    memory_service.search.return_value = [
        strong_memory,
        weak_memory,
    ]

    results = await retriever.retrieve(
        "software developer"
    )

    assert [memory.id for memory in results] == ["strong"]


@pytest.mark.asyncio
async def test_retriever_filters_before_applying_result_limit(
    memory_service,
):
    config = MemoryArchitectureConfig(
        retrieval_min_relevance=0.7,
    )

    retriever = MemoryRetriever(
        memory_service,
        max_results=1,
        config=config,
    )

    memory_service.search.return_value = [
        make_memory(
            "weak",
            "I build software products.",
        ),
        make_memory(
            "strong",
            "I am a software developer.",
        ),
    ]

    results = await retriever.retrieve(
        "software developer"
    )

    assert len(results) == 1
    assert results[0].id == "strong"


@pytest.mark.asyncio
async def test_default_thresholds_preserve_retrieval_behavior(
    memory_service,
):
    retriever = MemoryRetriever(memory_service)

    memory_service.search.return_value = [
        make_memory(
            "matching",
            "I am a software developer.",
        ),
        make_memory(
            "unrelated",
            "I enjoy cooking at home.",
        ),
    ]

    results = await retriever.retrieve(
        "software developer"
    )

    assert [memory.id for memory in results] == ["matching"]


@pytest.mark.asyncio
async def test_retriever_uses_custom_importance_and_score_thresholds(
    memory_service,
):
    config = MemoryArchitectureConfig(
        retrieval_min_importance=0.6,
        retrieval_min_relevance=0.5,
        retrieval_min_combined_score=0.5,
    )

    retriever = MemoryRetriever(
        memory_service,
        config=config,
    )

    memory_service.search.return_value = [
        make_memory(
            "low-importance",
            "I am a software developer.",
            importance=0.4,
        ),
        make_memory(
            "accepted",
            "I am a software developer.",
            importance=0.8,
        ),
    ]

    results = await retriever.retrieve(
        "software developer"
    )

    assert [memory.id for memory in results] == ["accepted"]