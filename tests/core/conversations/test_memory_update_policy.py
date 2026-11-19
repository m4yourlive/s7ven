
"""Tests for memory update decision policy."""

import pytest

from seven.core.conversations.memory_update_policy import (
    MemoryUpdateAction,
    MemoryUpdateDecision,
    MemoryUpdatePolicy,
)


@pytest.fixture
def policy():
    return MemoryUpdatePolicy()


def test_identical_information_reinforces_existing_memory(policy):
    decision = policy.evaluate(
        existing_content="I prefer concise responses.",
        incoming_content="  i prefer concise responses.  ",
        existing_confidence=0.8,
        incoming_confidence=0.9,
    )

    assert decision.action == MemoryUpdateAction.REINFORCE
    assert decision.confidence == 0.9


def test_higher_confidence_information_updates_memory(policy):
    decision = policy.evaluate(
        existing_content="I work as a designer.",
        incoming_content="I work as a software developer.",
        existing_confidence=0.6,
        incoming_confidence=0.9,
    )

    assert decision.action == MemoryUpdateAction.UPDATE
    assert decision.confidence == 0.9


def test_lower_confidence_information_keeps_existing_memory(policy):
    decision = policy.evaluate(
        existing_content="I work as a software developer.",
        incoming_content="I work as a designer.",
        existing_confidence=0.9,
        incoming_confidence=0.6,
    )

    assert decision.action == MemoryUpdateAction.KEEP
    assert decision.confidence == 0.9


def test_similar_confidence_conflict_requires_review(policy):
    decision = policy.evaluate(
        existing_content="I prefer dark mode.",
        incoming_content="I prefer light mode.",
        existing_confidence=0.8,
        incoming_confidence=0.82,
    )

    assert decision.action == MemoryUpdateAction.REVIEW


def test_decision_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        MemoryUpdateDecision(
            action=MemoryUpdateAction.KEEP,
            reason="Invalid confidence",
            confidence=1.5,
        )


def test_policy_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        MemoryUpdatePolicy(confidence_margin=1.5)


def test_evaluate_rejects_empty_content(policy):
    with pytest.raises(ValueError):
        policy.evaluate(
            existing_content=" ",
            incoming_content="New information",
            existing_confidence=0.8,
            incoming_confidence=0.9,
        )


def test_evaluate_rejects_invalid_confidence(policy):
    with pytest.raises(ValueError):
        policy.evaluate(
            existing_content="Existing information",
            incoming_content="New information",
            existing_confidence=0.8,
            incoming_confidence=1.5,
        )
