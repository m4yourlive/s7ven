"""Tests for context-aware memory scoring."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from seven.core.conversations import (
    MemoryArchitectureConfig,
    MemoryScorer,
    MemoryScoringContext,
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


def test_context_defaults():
    context = MemoryScoringContext()

    assert context.conversation_id is None
    assert context.preferred_categories == ()
    assert context.reference_time is None
    assert context.recency_half_life_days is None


def test_context_normalizes_conversation_id():
    context = MemoryScoringContext(
        conversation_id=" conversation-001 "
    )

    assert context.conversation_id == "conversation-001"


def test_context_normalizes_categories():
    context = MemoryScoringContext(
        preferred_categories=(
            "GOAL",
            "project",
            "goal",
        )
    )

    assert context.preferred_categories == (
        "goal",
        "project",
    )


def test_context_accepts_category_list():
    context = MemoryScoringContext(
        preferred_categories=["goal", "project"]
    )

    assert context.preferred_categories == ("goal", "project")


def test_context_accepts_custom_recency_half_life():
    context = MemoryScoringContext(
        recency_half_life_days=14,
    )

    assert context.recency_half_life_days == 14.0


def test_context_rejects_invalid_conversation_id():
    with pytest.raises(ValueError):
        MemoryScoringContext(conversation_id="   ")


def test_context_rejects_non_string_conversation_id():
    with pytest.raises(TypeError):
        MemoryScoringContext(conversation_id=123)


def test_context_rejects_invalid_category_collection():
    with pytest.raises(TypeError):
        MemoryScoringContext(
            preferred_categories="project"
        )


def test_context_rejects_non_datetime_reference_time():
    with pytest.raises(TypeError):
        MemoryScoringContext(
            reference_time="2026-01-01"
        )


@pytest.mark.parametrize(
    "half_life",
    [0, -1, float("nan"), float("inf")],
)
def test_context_rejects_invalid_recency_half_life(half_life):
    with pytest.raises(ValueError):
        MemoryScoringContext(
            recency_half_life_days=half_life
        )


def test_context_rejects_non_numeric_recency_half_life():
    with pytest.raises(TypeError):
        MemoryScoringContext(
            recency_half_life_days="30"
        )


def test_context_is_immutable():
    context = MemoryScoringContext()

    with pytest.raises(AttributeError):
        context.conversation_id = "conversation-002"


def test_context_category_affinity(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        preferred_categories=("project",),
        reference_time=reference_time,
    )

    matching_memory = make_memory(
        memory_id="matching",
        category="project",
        created_at=reference_time,
    )

    non_matching_memory = make_memory(
        memory_id="non-matching",
        category="personal",
        created_at=reference_time,
    )

    matching_score = scorer.score(
        "software developer",
        matching_memory,
        context,
    )

    non_matching_score = scorer.score(
        "software developer",
        non_matching_memory,
        context,
    )

    assert (
        matching_score.relevance_score
        > non_matching_score.relevance_score
    )


def test_context_conversation_affinity(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        conversation_id="conversation-001",
        reference_time=reference_time,
    )

    matching_memory = make_memory(
        memory_id="matching",
        conversation_id="conversation-001",
        created_at=reference_time,
    )

    non_matching_memory = make_memory(
        memory_id="non-matching",
        conversation_id="conversation-002",
        created_at=reference_time,
    )

    matching_score = scorer.score(
        "software developer",
        matching_memory,
        context,
    )

    non_matching_score = scorer.score(
        "software developer",
        non_matching_memory,
        context,
    )

    assert (
        matching_score.relevance_score
        > non_matching_score.relevance_score
    )


def test_context_recency_favors_newer_memories(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        reference_time=reference_time,
        recency_half_life_days=30,
    )

    recent_memory = make_memory(
        memory_id="recent",
        created_at=reference_time,
    )

    older_memory = make_memory(
        memory_id="older",
        created_at=reference_time - timedelta(days=60),
    )

    recent_score = scorer.score(
        "software developer",
        recent_memory,
        context,
    )

    older_score = scorer.score(
        "software developer",
        older_memory,
        context,
    )

    assert recent_score.relevance_score > older_score.relevance_score
    assert recent_score.combined_score > older_score.combined_score


def test_recency_half_life_reaches_half_after_one_period(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        reference_time=reference_time,
        recency_half_life_days=30,
    )

    memory = make_memory(
        created_at=reference_time - timedelta(days=30),
    )

    result = scorer.score(
        "software developer",
        memory,
        context,
    )

    assert 0.0 <= result.relevance_score <= 1.0

    recency = scorer._calculate_recency(
        memory.created_at,
        context,
    )

    assert recency == pytest.approx(0.5)


def test_future_memory_has_maximum_recency(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        reference_time=reference_time,
    )

    future_memory = make_memory(
        created_at=reference_time + timedelta(days=2),
    )

    recency = scorer._calculate_recency(
        future_memory.created_at,
        context,
    )

    assert recency == 1.0


def test_context_supports_naive_database_timestamps(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    naive_created_at = datetime(
        2026,
        11,
        17,
    )

    context = MemoryScoringContext(
        reference_time=reference_time,
    )

    memory = make_memory(
        created_at=naive_created_at,
    )

    result = scorer.score(
        "software developer",
        memory,
        context,
    )

    assert 0.0 <= result.relevance_score <= 1.0


def test_context_without_metadata_preserves_lexical_score(scorer):
    memory = make_memory()

    lexical = scorer.calculate_relevance(
        "software developer",
        memory.content,
    )

    result = scorer.score(
        "software developer",
        memory,
    )

    assert result.relevance_score == lexical.relevance_score


def test_context_without_matching_conversation_is_valid(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        conversation_id="conversation-999",
        reference_time=reference_time,
    )

    memory = make_memory(
        conversation_id="conversation-001",
        created_at=reference_time,
    )

    result = scorer.score(
        "software developer",
        memory,
        context,
    )

    assert 0.0 <= result.relevance_score <= 1.0


def test_rank_uses_contextual_scores(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        preferred_categories=("project",),
        reference_time=reference_time,
    )

    memories = [
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

    results = scorer.rank(
        "software developer",
        memories,
        context=context,
    )

    assert results[0].memory_id == "project"


def test_rank_respects_context_and_limit(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        preferred_categories=("project",),
        reference_time=reference_time,
    )

    memories = [
        make_memory(
            memory_id="one",
            category="personal",
            created_at=reference_time,
        ),
        make_memory(
            memory_id="two",
            category="project",
            created_at=reference_time,
        ),
        make_memory(
            memory_id="three",
            category="goal",
            created_at=reference_time,
        ),
    ]

    results = scorer.rank(
        "software developer",
        memories,
        limit=2,
        context=context,
    )

    assert len(results) == 2
    assert results[0].memory_id == "two"


def test_score_rejects_invalid_context(scorer):
    with pytest.raises(TypeError):
        scorer.score(
            "software developer",
            make_memory(),
            context="invalid",
        )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_lexical_weight", 0.8),
        ("context_category_weight", -0.1),
        ("context_conversation_weight", 1.1),
        ("context_recency_weight", float("nan")),
    ],
)
def test_config_rejects_invalid_context_weights(field, value):
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(**{field: value})

def test_config_requires_context_weights_to_sum_to_one():
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            context_lexical_weight=0.5,
            context_category_weight=0.1,
            context_conversation_weight=0.1,
            context_recency_weight=0.1,
        )

def test_config_accepts_custom_context_weights():
    config = MemoryArchitectureConfig(
        context_lexical_weight=0.6,
        context_category_weight=0.2,
        context_conversation_weight=0.1,
        context_recency_weight=0.1,
        recency_half_life_days=14,
    )

    assert config.context_lexical_weight == 0.6
    assert config.context_category_weight == 0.2
    assert config.context_conversation_weight == 0.1
    assert config.context_recency_weight == 0.1
    assert config.recency_half_life_days == 14.0

@pytest.mark.parametrize(
    "half_life",
    [0, -1, float("nan"), float("inf")],
)
def test_config_rejects_invalid_recency_half_life(half_life):
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            recency_half_life_days=half_life
        )

def test_config_rejects_non_numeric_recency_half_life():
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(
            recency_half_life_days="30"
        )