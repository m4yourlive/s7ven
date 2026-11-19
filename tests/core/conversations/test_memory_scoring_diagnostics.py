"""Tests for memory scoring transparency and diagnostics."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from seven.core.conversations import (
    MemoryScoreBreakdown,
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
    """Create a lightweight memory record."""
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category=category,
        importance=importance,
        conversation_id=conversation_id,
        created_at=created_at,
    )


def test_explain_returns_score_breakdown(scorer):
    breakdown = scorer.explain(
        "software developer",
        make_memory(),
    )

    assert isinstance(breakdown, MemoryScoreBreakdown)


def test_explain_returns_all_score_components(scorer):
    breakdown = scorer.explain(
        "software developer",
        make_memory(),
    )

    assert breakdown.lexical_score > 0.0
    assert breakdown.category_score == 0.0
    assert breakdown.conversation_score == 0.0
    assert breakdown.recency_score == 0.0
    assert breakdown.contextual_score == breakdown.lexical_score
    assert breakdown.importance_score == 0.8
    assert 0.0 <= breakdown.combined_score <= 1.0


def test_explain_calculates_category_match(scorer):
    context = MemoryScoringContext(
        preferred_categories=("project",),
    )

    breakdown = scorer.explain(
        "software developer",
        make_memory(category="project"),
        context,
    )

    assert breakdown.category_score == 1.0


def test_explain_detects_category_mismatch(scorer):
    context = MemoryScoringContext(
        preferred_categories=("project",),
    )

    breakdown = scorer.explain(
        "software developer",
        make_memory(category="personal"),
        context,
    )

    assert breakdown.category_score == 0.0


def test_explain_calculates_conversation_match(scorer):
    context = MemoryScoringContext(
        conversation_id="conversation-001",
    )

    breakdown = scorer.explain(
        "software developer",
        make_memory(
            conversation_id="conversation-001",
        ),
        context,
    )

    assert breakdown.conversation_score == 1.0


def test_explain_detects_conversation_mismatch(scorer):
    context = MemoryScoringContext(
        conversation_id="conversation-002",
    )

    breakdown = scorer.explain(
        "software developer",
        make_memory(
            conversation_id="conversation-001",
        ),
        context,
    )

    assert breakdown.conversation_score == 0.0


def test_explain_calculates_recency_component(scorer):
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

    breakdown = scorer.explain(
        "software developer",
        memory,
        context,
    )

    assert breakdown.recency_score == pytest.approx(0.5)


def test_explain_uses_all_context_components(scorer):
    reference_time = datetime(
        2026,
        11,
        18,
        tzinfo=timezone.utc,
    )

    context = MemoryScoringContext(
        conversation_id="conversation-001",
        preferred_categories=("project",),
        reference_time=reference_time,
    )

    memory = make_memory(
        category="project",
        conversation_id="conversation-001",
        created_at=reference_time,
    )

    breakdown = scorer.explain(
        "software developer",
        memory,
        context,
    )

    assert breakdown.category_score == 1.0
    assert breakdown.conversation_score == 1.0
    assert breakdown.recency_score == 1.0
    assert breakdown.contextual_score >= breakdown.lexical_score


def test_explain_combined_score_uses_configured_weights(scorer):
    memory = make_memory(importance=0.8)

    breakdown = scorer.explain(
        "software developer",
        memory,
    )

    expected = (
        breakdown.contextual_score * 0.7
        + breakdown.importance_score * 0.3
    )

    assert breakdown.combined_score == pytest.approx(expected)


def test_score_and_explain_return_consistent_results(scorer):
    memory = make_memory()

    breakdown = scorer.explain(
        "software developer",
        memory,
    )

    match = scorer.score(
        "software developer",
        memory,
    )

    assert match.relevance_score == breakdown.contextual_score
    assert match.combined_score == breakdown.combined_score
    assert match.importance == breakdown.importance_score


def test_explain_supports_memory_without_optional_fields(scorer):
    memory = SimpleNamespace(
        id="memory-001",
        content="I am a software developer.",
        category="personal",
        importance=0.8,
    )

    breakdown = scorer.explain(
        "software developer",
        memory,
    )

    assert breakdown.lexical_score > 0.0
    assert breakdown.recency_score == 0.0
    assert breakdown.conversation_score == 0.0


def test_explain_rejects_invalid_context(scorer):
    with pytest.raises(TypeError):
        scorer.explain(
            "software developer",
            make_memory(),
            context="invalid",
        )


def test_score_breakdown_is_immutable(scorer):
    breakdown = scorer.explain(
        "software developer",
        make_memory(),
    )

    with pytest.raises(AttributeError):
        breakdown.combined_score = 0.5


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ],
)
def test_score_breakdown_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        MemoryScoreBreakdown(
            lexical_score=value,
            category_score=0.0,
            conversation_score=0.0,
            recency_score=0.0,
            contextual_score=0.5,
            importance_score=0.8,
            combined_score=0.5,
        )


def test_score_breakdown_rejects_boolean_values():
    with pytest.raises(TypeError):
        MemoryScoreBreakdown(
            lexical_score=True,
            category_score=0.0,
            conversation_score=0.0,
            recency_score=0.0,
            contextual_score=0.5,
            importance_score=0.8,
            combined_score=0.5,
        )


def test_score_breakdown_rejects_non_numeric_values():
    with pytest.raises(TypeError):
        MemoryScoreBreakdown(
            lexical_score="high",
            category_score=0.0,
            conversation_score=0.0,
            recency_score=0.0,
            contextual_score=0.5,
            importance_score=0.8,
            combined_score=0.5,
        )