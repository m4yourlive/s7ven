
"""Tests for profile and memory context composition."""

import pytest

from seven.core.conversations.profile_context_builder import (
    ProfileContextBuilder,
)
from seven.core.conversations.profile_memory_context_composer import (
    ProfileMemoryContextComposer,
)
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)

@pytest.fixture
def profile():
    return UserProfile(
        attributes=(
            UserProfileAttribute(
                key="occupation",
                value="developer",
            ),
        ),
        preferences=(
            UserPreference(
                key="communication",
                value="concise",
            ),
        ),
    )

def test_compose_memory_and_profile_context(profile):
    composer = ProfileMemoryContextComposer()

    result = composer.compose(
        "Relevant memory: The user is building S7VEN.",
        profile,
    )

    assert "Relevant memory:" in result
    assert "User profile:" in result
    assert "occupation: developer" in result
    assert "User preferences:" in result
    assert "communication: concise" in result

def test_memory_context_appears_before_profile_context(profile):
    composer = ProfileMemoryContextComposer()

    result = composer.compose(
        "Memory context",
        profile,
    )

    assert result.index("Memory context") < result.index(
        "User profile:"
    )

def test_compose_memory_without_profile():
    composer = ProfileMemoryContextComposer()

    result = composer.compose("Memory context")

    assert result == "Memory context"

def test_compose_profile_without_memory(profile):
    composer = ProfileMemoryContextComposer()

    result = composer.compose("", profile)

    assert "User profile:" in result
    assert "occupation: developer" in result

def test_compose_empty_contexts():
    composer = ProfileMemoryContextComposer()

    assert composer.compose("", UserProfile()) == ""

def test_compose_strips_memory_context(profile):
    composer = ProfileMemoryContextComposer()

    result = composer.compose("  Memory context  ", profile)

    assert result.startswith("Memory context")

def test_compose_respects_maximum_context_length(profile):
    composer = ProfileMemoryContextComposer(
        max_context_chars=100,
    )

    result = composer.compose(
        "Memory context " * 10,
        profile,
    )

    assert len(result) <= 100

def test_memory_context_has_priority_when_limit_is_reached(
    profile,
):
    memory_context = "M" * 100
    composer = ProfileMemoryContextComposer(
        max_context_chars=50,
    )

    result = composer.compose(memory_context, profile)

    assert len(result) <= 50
    assert result == memory_context[:47] + "..."

def test_profile_context_is_truncated_when_space_is_limited(
    profile,
):
    composer = ProfileMemoryContextComposer(
        max_context_chars=100,
    )

    result = composer.compose("Memory", profile)

    assert len(result) <= 100
    assert result.startswith("Memory")

def test_custom_profile_builder_is_used(profile):
    builder = ProfileContextBuilder(
        min_confidence=1.0,
    )
    composer = ProfileMemoryContextComposer(
        profile_builder=builder,
    )

    result = composer.compose("", profile)

    assert "occupation: developer" in result
    assert "communication: concise" in result

def test_rejects_invalid_memory_context(profile):
    composer = ProfileMemoryContextComposer()

    with pytest.raises(TypeError):
        composer.compose(123, profile)

def test_rejects_invalid_profile():
    composer = ProfileMemoryContextComposer()

    with pytest.raises(TypeError):
        composer.compose("Memory", "invalid")

def test_rejects_invalid_profile_builder():
    with pytest.raises(TypeError):
        ProfileMemoryContextComposer(
            profile_builder=object(),
        )

@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_context_limit(value):
    with pytest.raises((TypeError, ValueError)):
        ProfileMemoryContextComposer(
            max_context_chars=value,
        )