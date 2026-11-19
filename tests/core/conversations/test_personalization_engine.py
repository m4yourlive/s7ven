
"""Tests for the rule-based personalization engine."""

import pytest

from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationRequest,
)
from seven.core.conversations.personalization_engine import (
    PersonalizationEngine,
)
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


@pytest.fixture
def engine() -> PersonalizationEngine:
    return PersonalizationEngine()


def test_personalize_generates_communication_directive(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="response_style",
                value="concise",
                confidence=0.95,
            ),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
        )
    )

    assert result.applied is True
    assert len(result.directives) == 1
    assert (
        result.directives[0].category
        is PersonalizationCategory.COMMUNICATION_STYLE
    )
    assert "concise" in result.directives[0].instruction
    assert "Personalization instructions:" in result.context


def test_personalize_recognizes_extracted_preference_phrases(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="prefers concise responses",
            ),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
        )
    )

    assert result.applied is True
    assert len(result.directives) == 1
    assert "concise" in result.directives[0].instruction


def test_personalize_generates_multiple_categories(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(key="response_length", value="detailed"),
            UserPreference(key="technical_depth", value="beginner"),
            UserPreference(key="formatting", value="step_by_step"),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
        )
    )

    categories = {item.category for item in result.directives}

    assert categories == {
        PersonalizationCategory.RESPONSE_LENGTH,
        PersonalizationCategory.TECHNICAL_DEPTH,
        PersonalizationCategory.FORMATTING,
    }


def test_personalize_uses_highest_confidence_per_category(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="response_style",
                value="casual",
                confidence=0.7,
            ),
            UserPreference(
                key="communication_style",
                value="formal",
                confidence=0.95,
            ),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
        )
    )

    assert len(result.directives) == 1
    assert result.directives[0].confidence == 0.95
    assert "formal" in result.directives[0].instruction


def test_personalize_filters_low_confidence_preferences(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="response_style",
                value="concise",
                confidence=0.5,
            ),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
            min_confidence=0.8,
        )
    )

    assert result.applied is False
    assert result.directives == ()
    assert result.context == ""


def test_personalize_ignores_unknown_preferences(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="unknown_setting",
                value="some_value",
            ),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
        )
    )

    assert result.applied is False


def test_personalize_generates_domain_context(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="interests",
                value="blockchain technology",
            ),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain smart contracts.",
            profile=profile,
        )
    )

    assert result.applied is True
    assert result.directives[0].category is (
        PersonalizationCategory.DOMAIN_CONTEXT
    )
    assert "blockchain technology" in result.directives[0].instruction


def test_personalize_respects_max_directives(
    engine: PersonalizationEngine,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(key="response_style", value="casual"),
            UserPreference(key="response_length", value="detailed"),
            UserPreference(key="technical_depth", value="beginner"),
        ),
    )

    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
            profile=profile,
            max_directives=2,
        )
    )

    assert len(result.directives) == 2


def test_personalize_returns_empty_result_without_profile(
    engine: PersonalizationEngine,
) -> None:
    result = engine.personalize(
        PersonalizationRequest(
            user_message="Explain Python.",
        )
    )

    assert result.applied is False
    assert result.directives == ()


def test_personalize_rejects_invalid_request(
    engine: PersonalizationEngine,
) -> None:
    with pytest.raises(TypeError):
        engine.personalize("invalid")  # type: ignore[arg-type]
