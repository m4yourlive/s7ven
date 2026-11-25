
"""Tests for the public personalization API."""

import seven.core.conversations as conversations


def test_personalization_public_api_exports() -> None:
    """Ensure all personalization components are publicly exported."""
    expected_exports = (
        "PersonalizationCategory",
        "PersonalizationDirective",
        "PersonalizationRequest",
        "PersonalizationResult",
        "PersonalizationEngine",
        "ResponseStylePersonalizer",
        "ContextAwarePersonalizer",
    )

    for name in expected_exports:
        assert hasattr(conversations, name)
        assert name in conversations.__all__


def test_personalization_public_api_imports() -> None:
    """Ensure personalization components can be imported directly."""
    from seven.core.conversations import (
        ContextAwarePersonalizer,
        PersonalizationCategory,
        PersonalizationDirective,
        PersonalizationEngine,
        PersonalizationRequest,
        PersonalizationResult,
        ResponseStylePersonalizer,
    )

    assert PersonalizationCategory is not None
    assert PersonalizationDirective is not None
    assert PersonalizationRequest is not None
    assert PersonalizationResult is not None
    assert PersonalizationEngine is not None
    assert ResponseStylePersonalizer is not None
    assert ContextAwarePersonalizer is not None


def test_public_api_has_no_duplicate_exports() -> None:
    """Ensure the public API contains no duplicate names."""
    assert len(conversations.__all__) == len(
        set(conversations.__all__)
    )


def test_all_public_exports_are_available() -> None:
    """Ensure every declared public export exists."""
    for name in conversations.__all__:
        assert hasattr(conversations, name), name


def test_existing_memory_and_profile_exports_remain_available() -> None:
    """Ensure personalization changes preserve existing public exports."""
    expected_exports = (
        "MemoryService",
        "MemoryRetriever",
        "MemoryConsolidator",
        "MemoryUpdateService",
        "MemoryConflictDetector",
        "UserProfile",
        "UserProfileService",
        "ProfileContextBuilder",
        "ProfileMemoryContextComposer",
    )

    for name in expected_exports:
        assert hasattr(conversations, name)
        assert name in conversations.__all__
