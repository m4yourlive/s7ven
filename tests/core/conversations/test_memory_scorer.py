"""Tests for memory relevance scoring and ranking."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from seven.core.conversations import (
    MemoryArchitectureConfig,
    MemoryRelevance,
    MemoryScorer,
)


@pytest.fixture
def scorer():
    return MemoryScorer()


def make_memory(
    memory_id="memory-001",
    content="I am a software developer.",
    category="personal",
    importance=0.8,
    conversation_id="conversation-001",
    created_at=None,
):
    """Create a lightweight memory record for scoring tests."""
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category=category,
        importance=importance,
        conversation_id=conversation_id,
        created_at=created_at,
    )


def test_tokenize_returns_unique_normalized_tokens(scorer):
    tokens = scorer.tokenize(
        "Software software DEVELOPMENT developer!"
    )

    assert tokens == frozenset(
        {"software", "development", "developer"}
    )


def test_tokenize_rejects_non_string(scorer):
    with pytest.raises(TypeError):
        scorer.tokenize(None)


def test_calculate_relevance_returns_structured_result(scorer):
    result = scorer.calculate_relevance(
        "software developer",
        "I am a software developer.",
    )

    assert isinstance(result, MemoryRelevance)
    assert result.token_coverage == 1.0
    assert result.phrase_match == 1.0
    assert 0.0 <= result.relevance_score <= 1.0


def test_exact_phrase_has_full_token_coverage(scorer):
    result = scorer.calculate_relevance(
        "software developer",
        "I am a software developer.",
    )

    assert result.token_coverage == 1.0


def test_exact_phrase_match_is_detected(scorer):
    result = scorer.calculate_relevance(
        "software developer",
        "I am a software developer.",
    )

    assert result.phrase_match == 1.0


def test_unrelated_query_has_zero_relevance(scorer):
    result = scorer.calculate_relevance(
        "blockchain finance",
        "I prefer minimal industrial design.",
    )

    assert result.token_coverage == 0.0
    assert result.token_precision == 0.0
    assert result.jaccard_similarity == 0.0
    assert result.phrase_match == 0.0
    assert result.relevance_score == 0.0


def test_partial_overlap_has_partial_relevance(scorer):
    result = scorer.calculate_relevance(
        "software developer",
        "I build software products.",
    )

    assert 0.0 < result.token_coverage <= 1.0
    assert 0.0 < result.relevance_score < 1.0


def test_relevance_score_is_normalized(scorer):
    result = scorer.calculate_relevance(
        "software developer",
        "I am a software developer.",
    )

    assert 0.0 <= result.relevance_score <= 1.0


def test_calculate_relevance_rejects_empty_query(scorer):
    with pytest.raises(ValueError):
        scorer.calculate_relevance(
            "   ",
            "I am a software developer.",
        )


def test_calculate_relevance_rejects_empty_content(scorer):
    with pytest.raises(ValueError):
        scorer.calculate_relevance(
            "software developer",
            "   ",
        )


def test_calculate_relevance_rejects_invalid_query_type(scorer):
    with pytest.raises(TypeError):
        scorer.calculate_relevance(
            None,
            "I am a software developer.",
        )


def test_calculate_relevance_rejects_invalid_content_type(scorer):
    with pytest.raises(TypeError):
        scorer.calculate_relevance(
            "software developer",
            None,
        )


def test_score_returns_search_match(scorer):
    match = scorer.score(
        "software developer",
        make_memory(),
    )

    assert match.memory_id == "memory-001"
    assert match.content == "I am a software developer."
    assert match.category == "personal"
    assert match.importance == 0.8
    assert match.relevance_score > 0.0
    assert 0.0 <= match.combined_score <= 1.0


def test_score_uses_configured_weighting():
    config = MemoryArchitectureConfig(
        relevance_weight=0.5,
        importance_weight=0.5,
    )
    scorer = MemoryScorer(config)

    memory = make_memory(
        importance=0.8,
    )

    match = scorer.score(
        "software developer",
        memory,
    )

    expected = (
        match.relevance_score * 0.5
        + 0.8 * 0.5
    )

    assert match.combined_score == pytest.approx(expected)


def test_score_preserves_conversation_metadata(scorer):
    created_at = datetime.now(timezone.utc)
    memory = make_memory(
        conversation_id="conversation-123",
        created_at=created_at,
    )

    match = scorer.score(
        "software developer",
        memory,
    )

    assert match.conversation_id == "conversation-123"
    assert match.created_at == created_at


@pytest.mark.parametrize(
    "importance",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_score_rejects_invalid_importance(scorer, importance):
    memory = make_memory(importance=importance)

    with pytest.raises(ValueError):
        scorer.score("software developer", memory)


@pytest.mark.parametrize(
    "importance",
    [True, "0.8"],
)
def test_score_rejects_non_numeric_importance(scorer, importance):
    memory = make_memory(importance=importance)

    with pytest.raises(TypeError):
        scorer.score("software developer", memory)


def test_score_rejects_missing_memory_fields(scorer):
    memory = SimpleNamespace(
        id="memory-001",
        content="A memory entry.",
    )

    with pytest.raises(TypeError):
        scorer.score("memory", memory)


def test_score_rejects_empty_memory_id(scorer):
    memory = make_memory(memory_id=" ")

    with pytest.raises(ValueError):
        scorer.score("software developer", memory)


def test_rank_returns_descending_combined_scores(scorer):
    memories = [
        make_memory(
            memory_id="low",
            content="I enjoy cooking at home.",
            importance=0.4,
        ),
        make_memory(
            memory_id="high",
            content="I am a software developer.",
            importance=0.8,
        ),
        make_memory(
            memory_id="medium",
            content="I build software products.",
            importance=0.6,
        ),
    ]

    results = scorer.rank(
        "software developer",
        memories,
    )

    assert len(results) == 3
    assert results[0].combined_score >= results[1].combined_score
    assert results[1].combined_score >= results[2].combined_score
    assert results[0].memory_id == "high"


def test_rank_respects_limit(scorer):
    memories = [
        make_memory(memory_id="first"),
        make_memory(
            memory_id="second",
            content="I develop Python applications.",
        ),
        make_memory(
            memory_id="third",
            content="I enjoy reading books.",
        ),
    ]

    results = scorer.rank(
        "software developer",
        memories,
        limit=2,
    )

    assert len(results) == 2


def test_rank_accepts_empty_memory_collection(scorer):
    results = scorer.rank(
        "software developer",
        [],
    )

    assert results == []


def test_rank_rejects_empty_query(scorer):
    with pytest.raises(ValueError):
        scorer.rank("", [])


def test_rank_rejects_invalid_limit(scorer):
    with pytest.raises(ValueError):
        scorer.rank("software developer", [], limit=0)


def test_rank_rejects_non_integer_limit(scorer):
    with pytest.raises(TypeError):
        scorer.rank("software developer", [], limit=2.5)


def test_scorer_rejects_invalid_config():
    with pytest.raises(TypeError):
        MemoryScorer(config="invalid")