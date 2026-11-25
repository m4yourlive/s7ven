
"""Tests for profile context building."""

import pytest

from seven.core.conversations.profile_context_builder import (
    ProfileContextBuilder,
)
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)

def test_build_empty_profile():
    builder = ProfileContextBuilder()

    assert builder.build(UserProfile()) == ""

def test_build_profile_attributes():
    builder = ProfileContextBuilder()
    profile = UserProfile(
        attributes=(
            UserProfileAttribute(
                key="occupation",
                value="developer",
            ),
            UserProfileAttribute(
                key="field_of_study",
                value="computer science",
            ),
        ),
    )

    result = builder.build(profile)

    assert "User profile:" in result
    assert "- occupation: developer" in result
    assert "- field of study: computer science" in result

def test_build_user_preferences():
    builder = ProfileContextBuilder()
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication_style",
                value="concise",
            ),
            UserPreference(
                key="dark_mode",
                value=True,
            ),
        ),
    )

    result = builder.build(profile)

    assert "User preferences:" in result
    assert "- communication style: concise" in result
    assert "- dark mode: yes" in result

def test_build_combines_attributes_and_preferences():
    builder = ProfileContextBuilder()
    profile = UserProfile(
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

    result = builder.build(profile)

    assert result.index("User profile:") < result.index(
        "User preferences:"
    )

def test_build_filters_low_confidence_entries():
    builder = ProfileContextBuilder(min_confidence=0.8)
    profile = UserProfile(
        attributes=(
            UserProfileAttribute(
                key="occupation",
                value="developer",
                confidence=0.95,
            ),
            UserProfileAttribute(
                key="location",
                value="Jakarta",
                confidence=0.5,
            ),
        ),
    )

    result = builder.build(profile)

    assert "occupation: developer" in result
    assert "location: Jakarta" not in result

def test_build_limits_attribute_count():
    builder = ProfileContextBuilder(max_attributes=1)
    profile = UserProfile(
        attributes=(
            UserProfileAttribute(key="occupation", value="developer"),
            UserProfileAttribute(key="location", value="Jakarta"),
        ),
    )

    result = builder.build(profile)

    assert "occupation: developer" in result
    assert "location: Jakarta" not in result

def test_build_limits_preference_count():
    builder = ProfileContextBuilder(max_preferences=1)
    profile = UserProfile(
        preferences=(
            UserPreference(key="communication", value="concise"),
            UserPreference(key="design", value="minimal"),
        ),
    )

    result = builder.build(profile)

    assert "communication: concise" in result
    assert "design: minimal" not in result

def test_build_truncates_long_context():
    builder = ProfileContextBuilder(max_context_chars=40)
    profile = UserProfile(
        attributes=(
            UserProfileAttribute(
                key="occupation",
                value="software developer with extensive experience",
            ),
        ),
    )

    result = builder.build(profile)

    assert len(result) <= 40
    assert result.endswith("...")

def test_build_preserves_boolean_false():
    builder = ProfileContextBuilder()
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="notifications",
                value=False,
            ),
        ),
    )

    assert "notifications: no" in builder.build(profile)

def test_build_rejects_invalid_profile():
    builder = ProfileContextBuilder()

    with pytest.raises(TypeError):
        builder.build("invalid")

@pytest.mark.parametrize(
    "value",
    [0, -1, 1.5, True],
)
def test_rejects_invalid_limits(value):
    with pytest.raises((TypeError, ValueError)):
        ProfileContextBuilder(max_attributes=value)

def test_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        ProfileContextBuilder(min_confidence=1.5)

def test_rejects_non_numeric_confidence():
    with pytest.raises(TypeError):
        ProfileContextBuilder(min_confidence="high")