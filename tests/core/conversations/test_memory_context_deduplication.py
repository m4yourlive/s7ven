
"""Tests for memory context deduplication."""

from types import SimpleNamespace

import pytest

from seven.core.conversations.memory_normalizer import (
    MemoryCandidateNormalizer,
)


def make_memory(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def test_deduplicate_memories_removes_exact_duplicates() -> None:
    first = make_memory("I prefer Python.")
    duplicate = make_memory("I prefer Python.")
    second = make_memory("I enjoy building applications.")

    result = MemoryCandidateNormalizer.deduplicate_memories(
        (first, duplicate, second)
    )

    assert result == (first, second)


def test_deduplicate_memories_uses_normalized_content() -> None:
    first = make_memory("I prefer Python.")
    duplicate = make_memory("  I PREFER PYTHON!  ")
    second = make_memory("I prefer JavaScript.")

    result = MemoryCandidateNormalizer.deduplicate_memories(
        (first, duplicate, second)
    )

    assert result == (first, second)


def test_deduplicate_memories_normalizes_unicode() -> None:
    first = make_memory("The user's preference is Python.")
    duplicate = make_memory("The user’s preference is Python.")

    result = MemoryCandidateNormalizer.deduplicate_memories(
        (first, duplicate)
    )

    assert result == (first,)


def test_deduplicate_memories_preserves_first_occurrence() -> None:
    first = make_memory("I work with Python.")
    second = make_memory("I study economics.")
    duplicate = make_memory("I WORK WITH PYTHON!")

    result = MemoryCandidateNormalizer.deduplicate_memories(
        (first, second, duplicate)
    )

    assert result == (first, second)


def test_deduplicate_memories_skips_empty_content() -> None:
    empty = make_memory("")
    whitespace = make_memory("   ")
    valid = make_memory("I prefer Python.")

    result = MemoryCandidateNormalizer.deduplicate_memories(
        (empty, whitespace, valid)
    )

    assert result == (valid,)


def test_deduplicate_memories_returns_empty_tuple_for_empty_input() -> None:
    result = MemoryCandidateNormalizer.deduplicate_memories(())

    assert result == ()


def test_deduplicate_memories_rejects_invalid_content() -> None:
    invalid = SimpleNamespace(content=123)

    with pytest.raises(TypeError):
        MemoryCandidateNormalizer.deduplicate_memories((invalid,))


def test_deduplicate_memories_rejects_missing_content() -> None:
    invalid = SimpleNamespace()

    with pytest.raises(TypeError):
        MemoryCandidateNormalizer.deduplicate_memories((invalid,))


def test_deduplicate_memories_preserves_memory_objects() -> None:
    first = make_memory("I prefer Python.")
    duplicate = make_memory("I PREFER PYTHON.")

    result = MemoryCandidateNormalizer.deduplicate_memories(
        (first, duplicate)
    )

    assert result[0] is first
