
"""Integration tests for the complete personalization workflow."""

import pytest

from seven.core.conversations.context_aware_personalizer import (
    ContextAwarePersonalizer,
)
from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationRequest,
)
from seven.core.conversations.personalization_engine import (
    PersonalizationEngine,
)
from seven.core.conversations.response_style_personalizer import (
    ResponseStylePersonalizer,
)
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


@pytest.fixture
def personalization_components():
    """Build the complete personalization workflow."""
    return (
        PersonalizationEngine(),
        ContextAwarePersonalizer(),
        ResponseStylePersonalizer(),
    )


def run_personalization(
    request: PersonalizationRequest,
    components,
):
    """Execute the complete personalization workflow."""
    engine, context_personalizer, style_personalizer = components

    initial_result = engine.personalize(request)

    context_result = context_personalizer.personalize(
        request,
        initial_result,
    )

    style_result = style_personalizer.personalize(context_result)

    domain_directives = tuple(
        directive
        for directive in context_result.directives
        if directive.category is PersonalizationCategory.DOMAIN_CONTEXT
    )

    return (
        initial_result,
        context_result,
        style_result,
        domain_directives,
    )


def test_complete_personalization_workflow(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="prefers concise responses",
                confidence=0.95,
            ),
            UserPreference(
                key="technical_depth",
                value="beginner",
                confidence=0.9,
            ),
            UserPreference(
                key="learning_style",
                value="practical",
                confidence=0.9,
            ),
            UserPreference(
                key="formatting",
                value="step_by_step",
                confidence=0.9,
            ),
            UserPreference(
                key="interests",
                value="blockchain technology",
                confidence=0.95,
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message=(
            "Teach me how to compare Python blockchain "
            "implementation options step by step."
        ),
        profile=profile,
    )

    initial, contextual, style, domain = run_personalization(
        request,
        personalization_components,
    )

    assert initial.applied is True
    assert len(initial.directives) == 5

    assert contextual.applied is True
    assert len(contextual.directives) == 5

    assert style.applied is True
    assert len(style.directives) == 4

    style_categories = {
        directive.category
        for directive in style.directives
    }

    assert style_categories == {
        PersonalizationCategory.COMMUNICATION_STYLE,
        PersonalizationCategory.TECHNICAL_DEPTH,
        PersonalizationCategory.LEARNING_STYLE,
        PersonalizationCategory.FORMATTING,
    }

    assert len(domain) == 1
    assert "blockchain technology" in domain[0].instruction

    assert "Response style instructions:" in style.context
    assert "concise" in style.context
    assert "fundamentals" in style.context


def test_global_preferences_survive_irrelevant_context(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="friendly",
            ),
            UserPreference(
                key="response_length",
                value="concise",
            ),
            UserPreference(
                key="technical_depth",
                value="beginner",
            ),
            UserPreference(
                key="learning_style",
                value="practical",
            ),
            UserPreference(
                key="formatting",
                value="bullet_points",
            ),
            UserPreference(
                key="interests",
                value="blockchain technology",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="What should I cook tonight?",
        profile=profile,
    )

    initial, contextual, style, domain = run_personalization(
        request,
        personalization_components,
    )

    assert initial.applied is True
    assert contextual.applied is True

    assert len(style.directives) == 2
    assert {
        directive.category
        for directive in style.directives
    } == {
        PersonalizationCategory.COMMUNICATION_STYLE,
        PersonalizationCategory.RESPONSE_LENGTH,
    }

    assert domain == ()


def test_memory_context_enables_relevant_personalization(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="technical_depth",
                value="beginner",
            ),
            UserPreference(
                key="learning_style",
                value="examples",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="Can you explain the next step?",
        profile=profile,
        memory_context="The user is learning Python programming.",
    )

    initial, contextual, style, _ = run_personalization(
        request,
        personalization_components,
    )

    assert initial.applied is True
    assert contextual.applied is True
    assert len(style.directives) == 2


def test_conversation_context_enables_domain_personalization(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="interests",
                value="blockchain technology",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="What are its security implications?",
        profile=profile,
        conversation_context=(
            "We are discussing blockchain infrastructure."
        ),
    )

    _, contextual, _, domain = run_personalization(
        request,
        personalization_components,
    )

    assert contextual.applied is True
    assert len(domain) == 1


def test_unrelated_domain_terms_do_not_match(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="interests",
                value="blockchain technology",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="Explain a chain reaction.",
        profile=profile,
    )

    _, contextual, _, domain = run_personalization(
        request,
        personalization_components,
    )

    assert contextual.applied is False
    assert domain == ()


def test_highest_confidence_preference_survives_full_workflow(
    personalization_components,
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

    request = PersonalizationRequest(
        user_message="Explain Python programming.",
        profile=profile,
    )

    initial, contextual, style, _ = run_personalization(
        request,
        personalization_components,
    )

    assert len(initial.directives) == 1
    assert initial.directives[0].confidence == 0.95

    assert contextual.applied is True
    assert len(style.directives) == 1
    assert "formal" in style.directives[0].instruction


def test_low_confidence_preferences_do_not_enter_workflow(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="concise",
                confidence=0.5,
            ),
            UserPreference(
                key="technical_depth",
                value="beginner",
                confidence=0.6,
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="Explain Python programming.",
        profile=profile,
        min_confidence=0.8,
    )

    initial, contextual, style, domain = run_personalization(
        request,
        personalization_components,
    )

    assert initial.applied is False
    assert contextual.applied is False
    assert style.applied is False
    assert domain == ()


def test_unknown_preferences_do_not_break_workflow(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="unknown_setting",
                value="some_value",
            ),
            UserPreference(
                key="communication",
                value="friendly",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="Hello.",
        profile=profile,
    )

    initial, contextual, style, _ = run_personalization(
        request,
        personalization_components,
    )

    assert len(initial.directives) == 1
    assert contextual.applied is True
    assert style.applied is True
    assert len(style.directives) == 1


def test_missing_profile_produces_no_personalization(
    personalization_components,
) -> None:
    request = PersonalizationRequest(
        user_message="Explain Python programming.",
    )

    initial, contextual, style, domain = run_personalization(
        request,
        personalization_components,
    )

    assert initial.applied is False
    assert contextual.applied is False
    assert style.applied is False
    assert domain == ()


def test_empty_personalization_does_not_create_context(
    personalization_components,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="unknown_setting",
                value="unsupported",
            ),
        ),
    )

    request = PersonalizationRequest(
        user_message="Explain Python.",
        profile=profile,
    )

    _, contextual, style, _ = run_personalization(
        request,
        personalization_components,
    )

    assert contextual.directives == ()
    assert contextual.context == ""
    assert style.directives == ()
    assert style.context == ""
