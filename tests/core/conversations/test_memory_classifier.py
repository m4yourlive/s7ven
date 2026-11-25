"""Tests for memory classification."""

import pytest

from seven.core.conversations import (
    MemoryClassification,
    MemoryClassifier,
    MemoryExtractor,
)

@pytest.fixture
def classifier():
    return MemoryClassifier()

@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "My long-term goal is to become a software developer.",
            "goal",
        ),
        (
            "I prefer minimal industrial design.",
            "preference",
        ),
        (
            "My project is a personal AI assistant.",
            "project",
        ),
        (
            "I study computer science at university.",
            "education",
        ),
        (
            "I work as a freelance developer.",
            "work",
        ),
        (
            "I am a software developer.",
            "personal",
        ),
        (
            "I have experience building Python applications.",
            "knowledge",
        ),
        (
            "The weather is warm today.",
            "general",
        ),
    ],
)
def test_classifier_returns_expected_category(
    classifier,
    content,
    expected,
):
    assert classifier.classify(content) == expected

def test_classifier_is_case_insensitive(classifier):
    assert classifier.classify(
        "MY GOAL IS TO BECOME A DEVELOPER."
    ) == "goal"

def test_classifier_respects_category_priority(classifier):
    assert classifier.classify(
        "My goal is to build a personal AI project."
    ) == "goal"

def test_classifier_prioritizes_knowledge_over_personal(classifier):
    assert classifier.classify(
        "I have experience building Python applications."
    ) == "knowledge"

def test_classifier_prioritizes_education_over_personal(classifier):
    assert classifier.classify(
        "I am studying computer science at university."
    ) == "education"

def test_classifier_does_not_match_partial_words(classifier):
    assert classifier.classify(
        "I am likely to become a software developer."
    ) == "personal"

def test_classifier_rejects_empty_content(classifier):
    with pytest.raises(ValueError):
        classifier.classify("   ")

def test_classifier_rejects_non_string_content(classifier):
    with pytest.raises(TypeError):
        classifier.classify(None)

def test_classify_with_evidence_returns_classification(
    classifier,
):
    result = classifier.classify_with_evidence(
        "I prefer minimal industrial design."
    )

    assert isinstance(result, MemoryClassification)
    assert result.category == "preference"
    assert result.confidence >= 0.82
    assert "i prefer" in result.evidence

def test_classification_includes_all_matching_evidence(
    classifier,
):
    result = classifier.classify_with_evidence(
        "I have experience building Python applications."
    )

    assert result.category == "knowledge"
    assert "i have experience" in result.evidence
    assert "experience" in result.evidence

def test_general_classification_has_no_evidence(classifier):
    result = classifier.classify_with_evidence(
        "The weather is warm today."
    )

    assert result.category == "general"
    assert result.confidence == 0.35
    assert result.evidence == ()

def test_classification_confidence_increases_with_evidence(
    classifier,
):
    single_match = classifier.classify_with_evidence(
        "I prefer clean interfaces."
    )

    multiple_matches = classifier.classify_with_evidence(
        "My preference is that I prefer minimal interfaces."
    )

    assert (
        multiple_matches.confidence
        >= single_match.confidence
    )

def test_classification_evidence_is_immutable(classifier):
    result = classifier.classify_with_evidence(
        "I prefer minimal interfaces."
    )

    assert isinstance(result.evidence, tuple)

def test_extractor_preserves_classification_metadata():
    extractor = MemoryExtractor()

    candidates = extractor.extract(
        "I have experience building Python applications."
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.category == "knowledge"
    assert candidate.classification_confidence >= 0.82
    assert "i have experience" in candidate.classification_evidence

def test_extractor_preserves_classification_evidence():
    extractor = MemoryExtractor()

    candidates = extractor.extract(
        "I prefer minimal industrial design."
    )

    assert len(candidates) == 1
    assert candidates[0].classification_evidence
    assert "i prefer" in candidates[0].classification_evidence

def test_memory_classification_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        MemoryClassification(
            category="goal",
            confidence=1.5,
            evidence=("goal",),
        )

def test_memory_classification_rejects_non_numeric_confidence():
    with pytest.raises(TypeError):
        MemoryClassification(
            category="goal",
            confidence="high",
            evidence=("goal",),
        )

def test_memory_classification_rejects_non_finite_confidence():
    with pytest.raises(ValueError):
        MemoryClassification(
            category="goal",
            confidence=float("nan"),
            evidence=("goal",),
        )

def test_memory_classification_normalizes_category():
    result = MemoryClassification(
        category="GOAL",
        confidence=0.9,
        evidence=["goal", "target"],
    )

    assert result.category == "goal"
    assert result.evidence == ("goal", "target")

def test_memory_classification_rejects_invalid_evidence():
    with pytest.raises(TypeError):
        MemoryClassification(
            category="goal",
            confidence=0.9,
            evidence=("goal", 123),
        )

def test_memory_classification_is_immutable():
    result = MemoryClassification(
        category="goal",
        confidence=0.9,
        evidence=("goal",),
    )

    with pytest.raises(AttributeError):
        result.category = "project"

def test_classifier_uses_default_general_fallback():
    classifier = MemoryClassifier()

    result = classifier.classify_with_evidence(
        "The weather is warm today."
    )

    assert result.category == "general"
    assert result.confidence == 0.35


def test_classifier_supports_custom_fallback():
    from seven.core.conversations import MemoryCategoryValidator

    validator = MemoryCategoryValidator(
        fallback_category="knowledge"
    )
    classifier = MemoryClassifier(
        category_validator=validator
    )

    result = classifier.classify_with_evidence(
        "The weather is warm today."
    )

    assert result.category == "knowledge"
    assert result.confidence == 0.35