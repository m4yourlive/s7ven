
"""Tests for memory conflict detection."""

import pytest

from seven.core.conversations.memory_conflict_detector import (
    MemoryConflictDetector,
    MemoryConflictType,
)

@pytest.fixture
def detector():
    return MemoryConflictDetector()

def test_identical_memories_have_no_conflict(detector):
    result = detector.detect(
        "I prefer concise responses.",
        "I prefer concise responses.",
    )

    assert result.is_conflict is False
    assert result.conflict_type == MemoryConflictType.NONE

def test_different_unrelated_memories_have_no_conflict(detector):
    result = detector.detect(
        "I enjoy playing football.",
        "I study computer science.",
    )

    assert result.is_conflict is False
    assert result.conflict_type == MemoryConflictType.NONE

def test_opposite_preferences_are_direct_conflicts(detector):
    result = detector.detect(
        "I prefer dark mode.",
        "I prefer light mode.",
    )

    assert result.is_conflict is True
    assert result.conflict_type == (
        MemoryConflictType.DIRECT_CONTRADICTION
    )

def test_negation_difference_is_detected(detector):
    result = detector.detect(
        "I like playing football.",
        "I do not like playing football.",
    )

    assert result.is_conflict is True
    assert result.conflict_type == (
        MemoryConflictType.DIRECT_CONTRADICTION
    )

def test_numeric_changes_are_detected(detector):
    result = detector.detect(
        "My daily study goal is 2 hours.",
        "My daily study goal is 4 hours.",
    )

    assert result.is_conflict is True
    assert result.conflict_type == MemoryConflictType.VALUE_CHANGE

def test_similar_context_with_different_details_is_possible_conflict(
    detector,
):
    result = detector.detect(
        "I work as a software developer.",
        "I work as a graphic designer.",
    )

    assert result.is_conflict is True
    assert result.conflict_type == MemoryConflictType.POSSIBLE_CONFLICT

def test_conflict_includes_shared_terms(detector):
    result = detector.detect(
        "I prefer dark mode.",
        "I prefer light mode.",
    )

    assert "mode" in result.shared_terms

def test_detector_rejects_empty_content(detector):
    with pytest.raises(ValueError):
        detector.detect(
            "",
            "I prefer dark mode.",
        )

def test_detector_rejects_invalid_content_type(detector):
    with pytest.raises(TypeError):
        detector.detect(
            None,
            "I prefer dark mode.",
        )

def test_detector_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        MemoryConflictDetector(minimum_overlap=1.5)

def test_detector_rejects_invalid_minimum_shared_terms():
    with pytest.raises(ValueError):
        MemoryConflictDetector(minimum_shared_terms=0)