"""Tests for memory candidate normalization and deduplication."""

import pytest

from seven.core.conversations import (
    MemoryCandidate,
    MemoryCandidateNormalizer,
)


def make_candidate(
    content: str,
    confidence: float = 0.8,
    importance: float = 0.75,
) -> MemoryCandidate:
    """Create a candidate for normalization tests."""
    return MemoryCandidate(
        content=content,
        category="preference",
        importance=importance,
        confidence=confidence,
    )


def test_normalize_text_case_insensitive():
    normalized = MemoryCandidateNormalizer.normalize_text(
        "I Prefer Minimal Design"
    )

    assert normalized == "i prefer minimal design"


def test_normalize_text_removes_punctuation_differences():
    first = MemoryCandidateNormalizer.normalize_text(
        "I prefer minimal design."
    )
    second = MemoryCandidateNormalizer.normalize_text(
        "I prefer minimal design!"
    )

    assert first == second


def test_normalize_text_collapses_whitespace():
    normalized = MemoryCandidateNormalizer.normalize_text(
        "I   prefer    minimal\n design."
    )

    assert normalized == "i prefer minimal design"


def test_normalize_text_normalizes_unicode_apostrophes():
    first = MemoryCandidateNormalizer.normalize_text(
        "I'm a software developer."
    )
    second = MemoryCandidateNormalizer.normalize_text(
        "I’m a software developer!"
    )

    assert first == second


def test_normalize_text_applies_unicode_compatibility():
    first = MemoryCandidateNormalizer.normalize_text(
        "I prefer minimal design."
    )
    second = MemoryCandidateNormalizer.normalize_text(
        "Ｉ ＰＲＥＦＥＲ ＭＩＮＩＭＡＬ ＤＥＳＩＧＮ！"
    )

    assert first == second


def test_normalize_text_rejects_non_string():
    with pytest.raises(TypeError):
        MemoryCandidateNormalizer.normalize_text(None)


def test_normalize_text_handles_empty_content():
    assert MemoryCandidateNormalizer.normalize_text("") == ""
    assert MemoryCandidateNormalizer.normalize_text("   ") == ""


def test_fingerprint_is_stable():
    first = MemoryCandidateNormalizer.fingerprint(
        "I prefer minimal design."
    )
    second = MemoryCandidateNormalizer.fingerprint(
        "I PREFER MINIMAL DESIGN!"
    )

    assert first == second


def test_fingerprint_returns_sha256_hex():
    fingerprint = MemoryCandidateNormalizer.fingerprint(
        "I prefer minimal design."
    )

    assert len(fingerprint) == 64
    assert all(
        character in "0123456789abcdef"
        for character in fingerprint
    )


def test_fingerprint_rejects_empty_content():
    with pytest.raises(ValueError):
        MemoryCandidateNormalizer.fingerprint("   ")


def test_fingerprint_rejects_non_string():
    with pytest.raises(TypeError):
        MemoryCandidateNormalizer.fingerprint(None)


def test_are_duplicates_matches_normalized_content():
    assert MemoryCandidateNormalizer.are_duplicates(
        "I prefer minimal design.",
        "I PREFER MINIMAL DESIGN!",
    )


def test_are_duplicates_rejects_different_content():
    assert not MemoryCandidateNormalizer.are_duplicates(
        "I prefer minimal design.",
        "I prefer modern design.",
    )


def test_are_duplicates_does_not_match_empty_content():
    assert not MemoryCandidateNormalizer.are_duplicates(
        "",
        "",
    )


def test_are_duplicates_rejects_non_string():
    with pytest.raises(TypeError):
        MemoryCandidateNormalizer.are_duplicates(
            None,
            "I prefer minimal design.",
        )


def test_deduplicate_candidates_removes_normalized_duplicates():
    candidates = [
        make_candidate("I prefer minimal design."),
        make_candidate("I PREFER MINIMAL DESIGN!"),
        make_candidate("I prefer modern design."),
    ]

    result = MemoryCandidateNormalizer.deduplicate_candidates(
        candidates
    )

    assert len(result) == 2
    assert result[0].content == "I prefer minimal design."
    assert result[1].content == "I prefer modern design."


def test_deduplicate_candidates_prefers_higher_confidence():
    candidates = [
        make_candidate(
            "I prefer minimal design.",
            confidence=0.75,
            importance=0.75,
        ),
        make_candidate(
            "I PREFER MINIMAL DESIGN!",
            confidence=0.95,
            importance=0.75,
        ),
    ]

    result = MemoryCandidateNormalizer.deduplicate_candidates(
        candidates
    )

    assert len(result) == 1
    assert result[0].confidence == 0.95


def test_deduplicate_candidates_uses_importance_as_tiebreaker():
    candidates = [
        make_candidate(
            "I prefer minimal design.",
            confidence=0.9,
            importance=0.7,
        ),
        make_candidate(
            "I PREFER MINIMAL DESIGN!",
            confidence=0.9,
            importance=0.8,
        ),
    ]

    result = MemoryCandidateNormalizer.deduplicate_candidates(
        candidates
    )

    assert len(result) == 1
    assert result[0].importance == 0.8


def test_deduplicate_candidates_preserves_first_position():
    candidates = [
        make_candidate("I prefer minimal design."),
        make_candidate("My goal is to become a developer."),
        make_candidate("I PREFER MINIMAL DESIGN!"),
    ]

    result = MemoryCandidateNormalizer.deduplicate_candidates(
        candidates
    )

    assert len(result) == 2
    assert result[0].category == "preference"
    assert result[1].content == "My goal is to become a developer."


def test_deduplicate_candidates_accepts_generators():
    candidates = (
        make_candidate(content)
        for content in (
            "I prefer minimal design.",
            "I PREFER MINIMAL DESIGN!",
        )
    )

    result = MemoryCandidateNormalizer.deduplicate_candidates(
        candidates
    )

    assert len(result) == 1


def test_deduplicate_candidates_ignores_empty_content():
    candidates = [
        make_candidate("   "),
        make_candidate("I prefer minimal design."),
    ]

    result = MemoryCandidateNormalizer.deduplicate_candidates(
        candidates
    )

    assert len(result) == 1
    assert result[0].content == "I prefer minimal design."


def test_deduplicate_candidates_rejects_invalid_items():
    with pytest.raises(TypeError):
        MemoryCandidateNormalizer.deduplicate_candidates(
            ["I prefer minimal design."]
        )