
"""Tests for user profile and preference contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from seven.core.conversations import (
    PreferenceUpdate,
    ProfileUpdate,
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)


def test_user_profile_attribute_normalizes_values():
    attribute = UserProfileAttribute(
        key="  Occupation ",
        value="  Student  ",
        source="  Explicit ",
    )

    assert attribute.key == "occupation"
    assert attribute.value == "Student"
    assert attribute.source == "explicit"
    assert attribute.confidence == 1.0


def test_user_preference_accepts_scalar_values():
    preference = UserPreference(
        key="dark_mode",
        value=True,
    )

    assert preference.key == "dark_mode"
    assert preference.value is True


@pytest.mark.parametrize("value", ["", "   "])
def test_profile_attribute_rejects_empty_key(value):
    with pytest.raises(ValueError):
        UserProfileAttribute(key=value, value="Student")


def test_profile_attribute_rejects_invalid_key_type():
    with pytest.raises(TypeError):
        UserProfileAttribute(key=123, value="Student")


@pytest.mark.parametrize("value", ["", "   "])
def test_profile_attribute_rejects_empty_string_value(value):
    with pytest.raises(ValueError):
        UserProfileAttribute(key="occupation", value=value)


@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_profile_attribute_rejects_unsupported_value(value):
    with pytest.raises(TypeError):
        UserProfileAttribute(key="occupation", value=value)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_profile_attribute_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        UserProfileAttribute(
            key="occupation",
            value="Student",
            confidence=confidence,
        )


@pytest.mark.parametrize("confidence", [True, "0.5", None])
def test_profile_attribute_rejects_non_numeric_confidence(confidence):
    with pytest.raises(TypeError):
        UserProfileAttribute(
            key="occupation",
            value="Student",
            confidence=confidence,
        )


def test_profile_attribute_rejects_invalid_source():
    with pytest.raises(ValueError):
        UserProfileAttribute(
            key="occupation",
            value="Student",
            source=" ",
        )


def test_profile_attribute_rejects_invalid_timestamp():
    with pytest.raises(TypeError):
        UserProfileAttribute(
            key="occupation",
            value="Student",
            updated_at="2026-01-01",
        )


def test_profile_attribute_is_immutable():
    attribute = UserProfileAttribute(
        key="occupation",
        value="Student",
    )

    with pytest.raises(FrozenInstanceError):
        attribute.value = "Developer"


def test_user_preference_normalizes_key_and_source():
    preference = UserPreference(
        key="  Theme ",
        value="  Dark ",
        source="  Explicit ",
    )

    assert preference.key == "theme"
    assert preference.value == "Dark"
    assert preference.source == "explicit"


def test_user_preference_accepts_numeric_values():
    preference = UserPreference(
        key="daily_goal",
        value=2.5,
    )

    assert preference.value == 2.5


def test_user_profile_accepts_attributes_and_preferences():
    profile = UserProfile(
        attributes=[
            UserProfileAttribute(
                key="occupation",
                value="Student",
            ),
        ],
        preferences=[
            UserPreference(
                key="theme",
                value="Dark",
            ),
        ],
    )

    assert len(profile.attributes) == 1
    assert len(profile.preferences) == 1
    assert profile.attributes[0].key == "occupation"
    assert profile.preferences[0].key == "theme"


def test_user_profile_defaults_to_empty_collections():
    profile = UserProfile()

    assert profile.attributes == ()
    assert profile.preferences == ()
    assert profile.updated_at is None


def test_user_profile_normalizes_collections_to_tuples():
    profile = UserProfile(
        attributes=[
            UserProfileAttribute(
                key="occupation",
                value="Student",
            ),
        ],
        preferences=[],
    )

    assert isinstance(profile.attributes, tuple)
    assert isinstance(profile.preferences, tuple)


def test_user_profile_rejects_invalid_attributes_collection():
    with pytest.raises(TypeError):
        UserProfile(attributes="invalid")


def test_user_profile_rejects_invalid_preferences_collection():
    with pytest.raises(TypeError):
        UserProfile(preferences="invalid")


def test_user_profile_rejects_invalid_attribute_items():
    with pytest.raises(TypeError):
        UserProfile(attributes=["occupation"])


def test_user_profile_rejects_invalid_preference_items():
    with pytest.raises(TypeError):
        UserProfile(preferences=["dark_mode"])


def test_user_profile_rejects_duplicate_attribute_keys():
    with pytest.raises(ValueError):
        UserProfile(
            attributes=[
                UserProfileAttribute(
                    key="occupation",
                    value="Student",
                ),
                UserProfileAttribute(
                    key="Occupation",
                    value="Developer",
                ),
            ],
        )


def test_user_profile_rejects_duplicate_preference_keys():
    with pytest.raises(ValueError):
        UserProfile(
            preferences=[
                UserPreference(key="theme", value="Dark"),
                UserPreference(key="Theme", value="Light"),
            ],
        )


def test_user_profile_rejects_invalid_updated_at():
    with pytest.raises(TypeError):
        UserProfile(updated_at="2026-01-01")


def test_user_profile_is_immutable():
    profile = UserProfile()

    with pytest.raises(FrozenInstanceError):
        profile.attributes = ()


def test_profile_update_converts_to_attribute():
    update = ProfileUpdate(
        key="  Occupation ",
        value="Student",
        confidence=0.9,
        source="explicit",
    )

    attribute = update.to_attribute()

    assert isinstance(attribute, UserProfileAttribute)
    assert attribute.key == "occupation"
    assert attribute.value == "Student"
    assert attribute.confidence == 0.9


def test_preference_update_converts_to_preference():
    update = PreferenceUpdate(
        key="  Theme ",
        value="Dark",
        confidence=0.95,
        source="explicit",
    )

    preference = update.to_preference()

    assert isinstance(preference, UserPreference)
    assert preference.key == "theme"
    assert preference.value == "Dark"
    assert preference.confidence == 0.95


def test_profile_update_preserves_timestamp():
    timestamp = datetime.now(timezone.utc)

    update = ProfileUpdate(
        key="occupation",
        value="Student",
        updated_at=timestamp,
    )

    assert update.to_attribute().updated_at == timestamp


def test_preference_update_preserves_timestamp():
    timestamp = datetime.now(timezone.utc)

    update = PreferenceUpdate(
        key="theme",
        value="Dark",
        updated_at=timestamp,
    )

    assert update.to_preference().updated_at == timestamp


def test_profile_update_is_immutable():
    update = ProfileUpdate(
        key="occupation",
        value="Student",
    )

    with pytest.raises(FrozenInstanceError):
        update.value = "Developer"


def test_preference_update_is_immutable():
    update = PreferenceUpdate(
        key="theme",
        value="Dark",
    )

    with pytest.raises(FrozenInstanceError):
        update.value = "Light"
