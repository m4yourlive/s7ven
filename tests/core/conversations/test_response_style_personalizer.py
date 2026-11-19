
"""Tests for response style personalization."""

import pytest

from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationDirective,
    PersonalizationResult,
)
from seven.core.conversations.response_style_personalizer import (
    ResponseStylePersonalizer,
)


@pytest.fixture
def personalizer() -> ResponseStylePersonalizer:
    return ResponseStylePersonalizer()


def test_personalize_selects_response_style_directives(
    personalizer: ResponseStylePersonalizer,
) -> None:
    directives = (
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use a friendly communication style.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.RESPONSE_LENGTH,
            instruction="Keep responses concise.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.FORMATTING,
            instruction="Prefer bullet points.",
        ),
    )

    result = personalizer.personalize(
        PersonalizationResult(
            directives=directives,
            applied=True,
        )
    )

    assert result.applied is True
    assert result.directives == directives
    assert "Response style instructions:" in result.context
    assert "Keep responses concise." in result.context


def test_personalize_excludes_domain_context(
    personalizer: ResponseStylePersonalizer,
) -> None:
    style = PersonalizationDirective(
        category=PersonalizationCategory.COMMUNICATION_STYLE,
        instruction="Use a friendly communication style.",
    )
    domain = PersonalizationDirective(
        category=PersonalizationCategory.DOMAIN_CONTEXT,
        instruction="Use blockchain as relevant domain context.",
    )

    result = personalizer.personalize(
        PersonalizationResult(
            directives=(style, domain),
            applied=True,
        )
    )

    assert result.directives == (style,)
    assert "blockchain" not in result.context


def test_personalize_selects_highest_confidence_per_category(
    personalizer: ResponseStylePersonalizer,
) -> None:
    directives = (
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use a casual communication style.",
            confidence=0.7,
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use a formal communication style.",
            confidence=0.95,
        ),
    )

    result = personalizer.personalize(
        PersonalizationResult(directives=directives)
    )

    assert len(result.directives) == 1
    assert result.directives[0].confidence == 0.95
    assert "formal" in result.directives[0].instruction


def test_personalize_uses_priority_when_confidence_matches(
    personalizer: ResponseStylePersonalizer,
) -> None:
    directives = (
        PersonalizationDirective(
            category=PersonalizationCategory.RESPONSE_LENGTH,
            instruction="Use detailed responses.",
            confidence=0.9,
            priority=30,
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.RESPONSE_LENGTH,
            instruction="Keep responses concise.",
            confidence=0.9,
            priority=10,
        ),
    )

    result = personalizer.personalize(
        PersonalizationResult(directives=directives)
    )

    assert len(result.directives) == 1
    assert "concise" in result.directives[0].instruction


def test_personalize_respects_max_directives() -> None:
    personalizer = ResponseStylePersonalizer(max_directives=2)

    directives = (
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use a friendly communication style.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.RESPONSE_LENGTH,
            instruction="Keep responses concise.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.FORMATTING,
            instruction="Prefer bullet points.",
        ),
    )

    result = personalizer.personalize(
        PersonalizationResult(directives=directives)
    )

    assert len(result.directives) == 2


def test_personalize_respects_max_context_chars() -> None:
    personalizer = ResponseStylePersonalizer(
        max_context_chars=80,
    )

    directives = (
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use a friendly communication style.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.RESPONSE_LENGTH,
            instruction="Keep responses concise and focused.",
        ),
    )

    result = personalizer.personalize(
        PersonalizationResult(directives=directives)
    )

    assert len(result.context) <= 80


def test_personalize_returns_empty_result_without_style_directives(
    personalizer: ResponseStylePersonalizer,
) -> None:
    result = personalizer.personalize(
        PersonalizationResult(
            directives=(
                PersonalizationDirective(
                    category=PersonalizationCategory.DOMAIN_CONTEXT,
                    instruction="Use Python as relevant domain context.",
                ),
            ),
            applied=True,
        )
    )

    assert result.applied is False
    assert result.directives == ()
    assert result.context == ""


def test_personalize_returns_empty_result_for_empty_input(
    personalizer: ResponseStylePersonalizer,
) -> None:
    result = personalizer.personalize(PersonalizationResult())

    assert result.applied is False
    assert result.directives == ()
    assert result.context == ""


def test_personalize_rejects_invalid_result(
    personalizer: ResponseStylePersonalizer,
) -> None:
    with pytest.raises(TypeError):
        personalizer.personalize("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [0, -1],
)
def test_constructor_rejects_invalid_context_limit(
    value: int,
) -> None:
    with pytest.raises(ValueError):
        ResponseStylePersonalizer(max_context_chars=value)


@pytest.mark.parametrize(
    "value",
    [0, -1],
)
def test_constructor_rejects_invalid_directive_limit(
    value: int,
) -> None:
    with pytest.raises(ValueError):
        ResponseStylePersonalizer(max_directives=value)


def test_constructor_rejects_boolean_limits() -> None:
    with pytest.raises(TypeError):
        ResponseStylePersonalizer(max_context_chars=True)
