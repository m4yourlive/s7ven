
"""Tests for the public memory update and consolidation API."""

import seven.core.conversations as conversations


def test_memory_update_public_api_exports() -> None:
    """Ensure memory update components are publicly exported."""
    expected_exports = (
        "MemoryConflict",
        "MemoryConflictType",
        "MemoryConflictDetector",
        "MemoryUpdateAction",
        "MemoryUpdateDecision",
        "MemoryUpdatePolicy",
        "MemoryUpdateResult",
        "MemoryUpdateService",
        "MemoryConsolidationAction",
        "MemoryConsolidationItem",
        "MemoryConsolidationResult",
    )

    for name in expected_exports:
        assert hasattr(conversations, name)
        assert name in conversations.__all__


def test_public_api_has_no_duplicate_exports() -> None:
    """Ensure the public API does not contain duplicate names."""
    assert len(conversations.__all__) == len(set(conversations.__all__))


def test_all_public_exports_are_available() -> None:
    """Ensure every declared public export exists."""
    for name in conversations.__all__:
        assert hasattr(conversations, name), name
