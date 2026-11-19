
"""Tests for personalization data contracts."""

import pytest

from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationDirective,
    PersonalizationRequest,
    PersonalizationResult,
)
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


def test_personalization_directive_normalizes_values() -> None:
    directive = PersonalizationDirective(
        category="communication_style",
        instruction="  Use concise responses.  ",
        source="  PROFILE  ",
        key="  response_style  ",
    )

    assert directive.category is PersonalizationCategory.COMMUNICATION_STYLE
    assert directive.instruction == "Use concise responses."
    assert directive.source == "profile"
    assert directive.key == "response_style"


def test_personalization_directive_rejects_invalid_category() -> None:
    with pytest.raises(ValueError):
        PersonalizationDirective(
            category="unknown",
            instruction="Use concise responses.",
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan")])
def test_personalization_directive_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use concise responses.",
            confidence=confidence,
        )


def test_personalization_directive_rejects_invalid_priority() -> None:
    with pytest.raises(ValueError):
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use concise responses.",
            priority=-1,
        )


def test_personalization_request_accepts_profile_and_context() -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="response_style",
                value="concise",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="Explain Python decorators.",
        profile=profile,
        memory_context="Previously discussed Python basics.",
        conversation_context="The user is learning Python.",
        min_confidence=0.8,
        max_directives=5,
    )

    assert request.profile is profile
    assert request.user_message == "Explain Python decorators."
    assert request.memory_context == (
        "Previously discussed Python basics."
    )
    assert request.conversation_context == (
        "The user is learning Python."
    )
    assert request.min_confidence == 0.8
    assert request.max_directives == 5


def test_personalization_request_rejects_empty_user_message() -> None:
    with pytest.raises(ValueError):
        PersonalizationRequest(user_message="  ")


def test_personalization_request_rejects_invalid_profile() -> None:
    with pytest.raises(TypeError):
        PersonalizationRequest(
            user_message="Explain Python.",
            profile="invalid",  # type: ignore[arg-type]
        )


def test_personalization_request_rejects_invalid_max_directives() -> None:
    with pytest.raises(ValueError):
        PersonalizationRequest(
            user_message="Explain Python.",
            max_directives=0,
        )


def test_personalization_result_normalizes_directives() -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.RESPONSE_LENGTH,
        instruction="Keep the answer concise.",
    )

    result = PersonalizationResult(
        directives=[directive],
        context="  Personalization context.  ",
        applied=True,
    )

    assert result.directives == (directive,)
    assert result.context == "Personalization context."
    assert result.applied is True


def test_personalization_result_rejects_applied_without_directives() -> None:
    with pytest.raises(ValueError):
        PersonalizationResult(applied=True)


def test_personalization_result_rejects_invalid_directives() -> None:
    with pytest.raises(TypeError):
        PersonalizationResult(
            directives=["invalid"],  # type: ignore[list-item]
        )


def test_personalization_result_defaults_to_not_applied() -> None:
    result = PersonalizationResult()

    assert result.directives == ()
    assert result.context == ""
    assert result.applied is False
    