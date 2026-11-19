"""Tests for automatic memory extraction."""

import pytest

from seven.core.conversations import MemoryExtractor


@pytest.fixture
def extractor():
    return MemoryExtractor()


def test_extract_personal_memory(extractor):
    candidates = extractor.extract(
        "I am a software developer."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "personal"
    assert candidates[0].content == "I am a software developer"


def test_extract_preference_memory(extractor):
    candidates = extractor.extract(
        "I prefer minimal industrial design."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "preference"


def test_extract_goal_memory(extractor):
    candidates = extractor.extract(
        "My goal is to become a software developer."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "goal"


def test_extract_project_memory(extractor):
    candidates = extractor.extract(
        "My project is a personal AI assistant."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "project"


def test_extract_work_memory(extractor):
    candidates = extractor.extract(
        "I work as a freelance developer."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "work"


def test_extract_education_memory(extractor):
    candidates = extractor.extract(
        "I study computer science at university."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "education"


def test_extract_multiple_candidates(extractor):
    text = (
        "I prefer minimal interfaces. "
        "My goal is to become a developer. "
        "My project is a personal assistant."
    )

    candidates = extractor.extract(text)

    assert len(candidates) == 3
    assert [item.category for item in candidates] == [
        "preference",
        "goal",
        "project",
    ]


def test_extract_removes_duplicate_candidates(extractor):
    text = (
        "I prefer minimal interfaces. "
        "I prefer minimal interfaces!"
    )

    candidates = extractor.extract(text)

    assert len(candidates) == 1


def test_extract_ignores_text_without_memory_signals(extractor):
    candidates = extractor.extract(
        "The weather is clear today."
    )

    assert candidates == []


def test_extract_ignores_empty_text(extractor):
    assert extractor.extract("") == []
    assert extractor.extract("   ") == []


def test_extract_normalizes_terminal_punctuation(extractor):
    candidates = extractor.extract(
        "I prefer clean and minimal interfaces."
    )

    assert candidates[0].content == (
        "I prefer clean and minimal interfaces"
    )


def test_extract_assigns_category_importance(extractor):
    candidates = extractor.extract(
        "My goal is to become a software developer."
    )

    assert candidates[0].importance == 0.85


def test_directive_increases_importance(extractor):
    candidates = extractor.extract(
        "Remember that I prefer minimal interfaces."
    )

    assert len(candidates) == 1
    assert candidates[0].importance == 0.85


def test_importance_never_exceeds_one(extractor):
    candidates = extractor.extract(
        "Remember that my goal is to become a developer."
    )

    assert len(candidates) == 1
    assert candidates[0].importance <= 1.0


def test_extract_rejects_non_string_input(extractor):
    with pytest.raises(TypeError):
        extractor.extract(None)


def test_constructor_rejects_invalid_minimum_length():
    with pytest.raises(ValueError):
        MemoryExtractor(min_length=0)


def test_constructor_rejects_invalid_maximum_length():
    with pytest.raises(ValueError):
        MemoryExtractor(min_length=100, max_length=50)


def test_constructor_rejects_non_integer_lengths():
    with pytest.raises(TypeError):
        MemoryExtractor(min_length=12.5)


def test_extractor_respects_minimum_length():
    extractor = MemoryExtractor(min_length=30)

    candidates = extractor.extract(
        "I prefer blue."
    )

    assert candidates == []


def test_extractor_respects_maximum_length():
    extractor = MemoryExtractor(max_length=30)

    text = (
        "I prefer interfaces that have many different "
        "components and complex visual arrangements."
    )

    assert extractor.extract(text) == []


def test_extractor_is_case_insensitive(extractor):
    candidates = extractor.extract(
        "MY GOAL IS TO BECOME A SOFTWARE DEVELOPER."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "goal"

def test_category_matching_does_not_match_partial_words(extractor):
    candidates = extractor.extract(
        "I am likely to become a better software developer."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "personal"


def test_preference_matching_supports_dislike(extractor):
    candidates = extractor.extract(
        "I dislike interfaces with excessive visual clutter."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "preference"

def test_extractor_rejects_vague_candidates(extractor):
    candidates = extractor.extract("I like stuff.")

    assert candidates == []

def test_extractor_rejects_speculative_candidates(extractor):
    candidates = extractor.extract(
        "Maybe I want to become a software developer."
    )

    assert candidates == []

def test_extractor_rejects_question_candidates(extractor):
    candidates = extractor.extract(
        "I want to become a software developer?"
    )

    assert candidates == []

def test_extracted_candidate_contains_confidence(extractor):
    candidates = extractor.extract(
        "I am a software developer."
    )

    assert len(candidates) == 1
    assert candidates[0].confidence >= 0.65
    assert candidates[0].confidence <= 1.0

def test_extractor_classifies_knowledge_memory(extractor):
    candidates = extractor.extract(
        "I have experience building Python applications."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "knowledge"

def test_extractor_accepts_injected_classifier():
    from seven.core.conversations import MemoryClassifier

    classifier = MemoryClassifier()
    extractor = MemoryExtractor(classifier=classifier)

    candidates = extractor.extract(
        "I know Python and JavaScript very well."
    )

    assert len(candidates) == 1
    assert candidates[0].category == "knowledge"

def test_extractor_rejects_invalid_classifier():
    with pytest.raises(TypeError):
        MemoryExtractor(classifier="invalid")