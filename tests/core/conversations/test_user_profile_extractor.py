"""Tests for structured user profile extraction."""

import pytest

from seven.core.conversations import UserProfileExtractor


@pytest.fixture
def extractor():
    return UserProfileExtractor()


def test_extract_empty_profile(extractor):
    assert extractor.extract("") .attributes == ()
    assert extractor.extract("   ").preferences == ()


def test_extract_rejects_non_string_input(extractor):
    with pytest.raises(TypeError):
        extractor.extract(None)


def test_extract_name(extractor):
    profile = extractor.extract("My name is Alex.")

    assert len(profile.attributes) == 1
    assert profile.attributes[0].key == "name"
    assert profile.attributes[0].value == "Alex"


def test_extract_age(extractor):
    profile = extractor.extract("I am 16 years old.")

    assert len(profile.attributes) == 1
    assert profile.attributes[0].key == "age"
    assert profile.attributes[0].value == 16


def test_extract_location(extractor):
    profile = extractor.extract("I live in Indonesia.")

    assert profile.attributes[0].key == "location"
    assert profile.attributes[0].value == "Indonesia"


def test_extract_occupation(extractor):
    profile = extractor.extract(
        "I work as a software developer."
    )

    assert profile.attributes[0].key == "occupation"
    assert profile.attributes[0].value == "a software developer"


def test_extract_field_of_study(extractor):
    profile = extractor.extract(
        "I study computer science."
    )

    assert profile.attributes[0].key == "field_of_study"
    assert profile.attributes[0].value == "computer science"


def test_extract_general_role(extractor):
    profile = extractor.extract(
        "I am a software developer."
    )

    assert profile.attributes[0].key == "role"
    assert profile.attributes[0].value == "software developer"


def test_extract_preference(extractor):
    profile = extractor.extract(
        "I prefer minimal industrial design."
    )

    assert len(profile.preferences) == 1
    assert profile.preferences[0].key == "preference_design"
    assert profile.preferences[0].value == (
        "prefers minimal industrial design"
    )


def test_extract_dislike_preference(extractor):
    profile = extractor.extract(
        "I dislike excessive visual clutter."
    )

    assert len(profile.preferences) == 1
    assert profile.preferences[0].value == (
        "dislikes excessive visual clutter"
    )


def test_extract_communication_preference(extractor):
    profile = extractor.extract(
        "I prefer concise responses."
    )

    assert profile.preferences[0].key == (
        "preference_communication"
    )


def test_extract_multiple_attributes_and_preferences(extractor):
    profile = extractor.extract(
        "My name is Alex. "
        "I am 16 years old. "
        "I prefer minimal industrial design."
    )

    assert len(profile.attributes) == 2
    assert len(profile.preferences) == 1


def test_extract_deduplicates_attributes(extractor):
    profile = extractor.extract(
        "My name is Alex. "
        "My name is Jordan."
    )

    assert len(profile.attributes) == 1
    assert profile.attributes[0].value == "Alex"


def test_extract_deduplicates_preferences(extractor):
    profile = extractor.extract(
        "I prefer minimal industrial design. "
        "I like minimal industrial design."
    )

    assert len(profile.preferences) == 1


def test_profile_attribute_has_confidence_and_source(extractor):
    profile = extractor.extract(
        "My name is Alex."
    )

    attribute = profile.attributes[0]

    assert attribute.confidence == 0.95
    assert attribute.source == "explicit"


def test_preference_has_confidence_and_source(extractor):
    profile = extractor.extract(
        "I prefer minimal industrial design."
    )

    preference = profile.preferences[0]

    assert preference.confidence == 0.90
    assert preference.source == "explicit"


def test_ignores_unrecognized_statements(extractor):
    profile = extractor.extract(
        "The weather is clear today."
    )

    assert profile.attributes == ()
    assert profile.preferences == ()


def test_ignores_unsupported_preference_pattern(extractor):
    profile = extractor.extract(
        "Maybe I prefer minimal industrial design."
    )

    assert profile.preferences == ()


def test_confidence_threshold_rejects_low_confidence():
    extractor = UserProfileExtractor(
        confidence_threshold=0.96
    )

    profile = extractor.extract(
        "My name is Alex. "
        "I prefer minimal industrial design."
    )

    assert profile.attributes == ()
    assert profile.preferences == ()


def test_constructor_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        UserProfileExtractor(confidence_threshold=1.5)


def test_constructor_rejects_non_numeric_threshold():
    with pytest.raises(TypeError):
        UserProfileExtractor(confidence_threshold="high")


def test_constructor_rejects_boolean_threshold():
    with pytest.raises(TypeError):
        UserProfileExtractor(confidence_threshold=True)
