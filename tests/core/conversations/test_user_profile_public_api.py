
"""Tests for the public user profile API."""

from seven.core.conversations import (
    PreferenceUpdate,
    ProfileContextBuilder,
    ProfileMemoryContextComposer,
    ProfileUpdate,
    UserPreference,
    UserProfile,
    UserProfileAttribute,
    UserProfileExtractor,
    UserProfileService,
)


def test_profile_contracts_are_publicly_exported():
    assert UserProfileAttribute is not None
    assert UserPreference is not None
    assert UserProfile is not None
    assert ProfileUpdate is not None
    assert PreferenceUpdate is not None


def test_profile_components_are_publicly_exported():
    assert UserProfileExtractor is not None
    assert UserProfileService is not None
    assert ProfileContextBuilder is not None
    assert ProfileMemoryContextComposer is not None


def test_profile_components_can_be_instantiated():
    extractor = UserProfileExtractor()
    service_builder = ProfileContextBuilder()
    composer = ProfileMemoryContextComposer()

    assert extractor is not None
    assert service_builder is not None
    assert composer is not None
