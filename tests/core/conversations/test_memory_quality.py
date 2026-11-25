"""Tests for memory extraction quality validation."""

import pytest

from seven.core.conversations import (
    MemoryCandidate,
    MemoryQualityValidator,
)

@pytest.fixture
def validator():
    return MemoryQualityValidator()

def make_candidate(
    content: str,
    category: str = "goal",
) -> MemoryCandidate:
    """Create a candidate for quality tests."""
    return MemoryCandidate(
        content=content,
        category=category,
        importance=0.8,
    )

def test_validator_accepts_clear_personal_information(validator):
    candidate = make_candidate(
        "I am a software developer.",
        category="personal",
    )

    confidence = validator.assess(candidate)

    assert confidence >= validator.min_confidence
    assert validator.is_acceptable(confidence) is True

def test_validator_accepts_clear_preferences(validator):
    candidate = make_candidate(
        "I prefer minimal industrial design.",
        category="preference",
    )

    assert validator.is_acceptable(
        validator.assess(candidate)
    )

def test_validator_accepts_explicit_directives(validator):
    candidate = make_candidate(
        "Remember that I prefer minimal interfaces.",
        category="preference",
    )

    confidence = validator.assess(candidate)

    assert confidence == 1.0
    assert validator.is_acceptable(confidence)

def test_validator_rejects_vague_information(validator):
    candidate = make_candidate(
        "I like stuff.",
        category="preference",
    )

    confidence = validator.assess(candidate)

    assert confidence < validator.min_confidence
    assert not validator.is_acceptable(confidence)

def test_validator_rejects_speculative_information(validator):
    candidate = make_candidate(
        "Maybe I want to become a software developer.",
        category="goal",
    )

    confidence = validator.assess(candidate)

    assert confidence < validator.min_confidence
    assert not validator.is_acceptable(confidence)

def test_validator_rejects_question_candidates(validator):
    candidate = make_candidate(
        "I want to become a software developer?",
        category="goal",
    )

    confidence = validator.assess(candidate)

    assert confidence < validator.min_confidence
    assert not validator.is_acceptable(confidence)

def test_validator_rejects_generic_category_candidates(validator):
    candidate = make_candidate(
        "The weather is generally warm today.",
        category="general",
    )

    confidence = validator.assess(candidate)

    assert not validator.is_acceptable(confidence)

def test_validator_returns_normalized_confidence(validator):
    candidate = make_candidate(
        "My long-term goal is to become a software developer.",
        category="goal",
    )

    confidence = validator.assess(candidate)

    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

def test_validator_supports_custom_threshold():
    validator = MemoryQualityValidator(min_confidence=0.9)

    candidate = make_candidate(
        "I prefer minimal interfaces.",
        category="preference",
    )

    confidence = validator.assess(candidate)

    assert confidence < 0.9
    assert not validator.is_acceptable(confidence)

def test_validator_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        MemoryQualityValidator(min_confidence=1.1)

def test_validator_rejects_non_numeric_threshold():
    with pytest.raises(TypeError):
        MemoryQualityValidator(min_confidence="0.5")

def test_validator_rejects_non_finite_threshold():
    with pytest.raises(ValueError):
        MemoryQualityValidator(min_confidence=float("nan"))

def test_validator_rejects_invalid_candidate():
    validator = MemoryQualityValidator()

    with pytest.raises(TypeError):
        validator.assess("I am a software developer.")

@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_validator_rejects_invalid_confidence_values(
    validator,
    confidence,
):
    with pytest.raises(ValueError):
        validator.is_acceptable(confidence)

def test_validator_rejects_non_numeric_confidence(validator):
    with pytest.raises(TypeError):
        validator.is_acceptable("0.8")

def test_memory_candidate_equality_ignores_confidence():
    first = MemoryCandidate(
        content="I prefer minimal interfaces.",
        category="preference",
        importance=0.75,
        confidence=0.8,
    )

    second = MemoryCandidate(
        content="I prefer minimal interfaces.",
        category="preference",
        importance=0.75,
        confidence=0.95,
    )

    assert first == second