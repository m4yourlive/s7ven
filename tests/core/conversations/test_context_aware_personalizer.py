
"""Tests for context-aware personalization."""

import pytest

from seven.core.conversations.context_aware_personalizer import (
    ContextAwarePersonalizer,
)
from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationDirective,
    PersonalizationRequest,
    PersonalizationResult,
)

@pytest.fixture
def personalizer() -> ContextAwarePersonalizer:
    return ContextAwarePersonalizer()

def test_personalize_keeps_global_style_directives(
    personalizer: ContextAwarePersonalizer,
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
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="What should I cook tonight?",
        ),
        PersonalizationResult(directives=directives),
    )

    assert result.applied is True
    assert result.directives == directives

def test_personalize_applies_technical_depth_to_technical_topics(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.TECHNICAL_DEPTH,
        instruction="Explain technical concepts from the fundamentals.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="Explain Python functions and their implementation.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is True
    assert result.directives == (directive,)

def test_personalize_excludes_technical_depth_from_unrelated_topics(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.TECHNICAL_DEPTH,
        instruction="Explain technical concepts from the fundamentals.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="What are some good dinner recipes?",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is False
    assert result.directives == ()

def test_personalize_applies_learning_style_to_learning_requests(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.LEARNING_STYLE,
        instruction="Use concrete examples to explain unfamiliar concepts.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="Teach me how to understand Python decorators.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is True
    assert result.directives == (directive,)

def test_personalize_excludes_learning_style_from_unrelated_requests(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.LEARNING_STYLE,
        instruction="Use practical exercises.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="What is the weather like?",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is False

def test_personalize_applies_formatting_to_structured_requests(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.FORMATTING,
        instruction="Use numbered lists for ordered instructions.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="Compare Python and JavaScript.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is True
    assert result.directives == (directive,)

def test_personalize_excludes_formatting_from_simple_requests(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.FORMATTING,
        instruction="Prefer bullet points.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="What is Python?",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is False

def test_personalize_matches_domain_in_user_message(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.DOMAIN_CONTEXT,
        instruction=(
            "Use blockchain technology as relevant domain context "
            "when answering related questions."
        ),
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="Explain blockchain smart contracts.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is True
    assert result.directives == (directive,)

def test_personalize_matches_domain_in_memory_context(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.DOMAIN_CONTEXT,
        instruction=(
            "Use Python programming as relevant domain context "
            "when answering related questions."
        ),
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="How should I approach this problem?",
            memory_context="The user has been learning Python programming.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is True

def test_personalize_matches_domain_in_conversation_context(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.DOMAIN_CONTEXT,
        instruction=(
            "Use blockchain as relevant domain context "
            "when answering related questions."
        ),
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="What about its security?",
            conversation_context="We are discussing blockchain infrastructure.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is True

def test_personalize_excludes_unrelated_domain_context(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.DOMAIN_CONTEXT,
        instruction=(
            "Use blockchain technology as relevant domain context "
            "when answering related questions."
        ),
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="Give me some dinner ideas.",
        ),
        PersonalizationResult(directives=(directive,)),
    )

    assert result.applied is False

def test_personalize_combines_relevant_categories(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directives = (
        PersonalizationDirective(
            category=PersonalizationCategory.COMMUNICATION_STYLE,
            instruction="Use a friendly communication style.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.TECHNICAL_DEPTH,
            instruction="Explain technical concepts from the fundamentals.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.LEARNING_STYLE,
            instruction="Use practical exercises.",
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.DOMAIN_CONTEXT,
            instruction=(
                "Use Python programming as relevant domain context."
            ),
        ),
        PersonalizationDirective(
            category=PersonalizationCategory.FORMATTING,
            instruction="Use numbered lists.",
        ),
    )

    result = personalizer.personalize(
        PersonalizationRequest(
            user_message="Teach me how to compare Python programming approaches.",
        ),
        PersonalizationResult(directives=directives),
    )

    assert result.applied is True
    assert len(result.directives) == 5

def test_personalize_returns_empty_result_without_directives(
    personalizer: ContextAwarePersonalizer,
) -> None:
    result = personalizer.personalize(
        PersonalizationRequest(user_message="Explain Python."),
        PersonalizationResult(),
    )

    assert result.applied is False
    assert result.directives == ()
    assert result.context == ""

def test_personalize_builds_context_from_selected_directives(
    personalizer: ContextAwarePersonalizer,
) -> None:
    directive = PersonalizationDirective(
        category=PersonalizationCategory.COMMUNICATION_STYLE,
        instruction="Use a friendly communication style.",
    )

    result = personalizer.personalize(
        PersonalizationRequest(user_message="Hello."),
        PersonalizationResult(directives=(directive,)),
    )

    assert "Context-aware personalization instructions:" in result.context
    assert directive.instruction in result.context

def test_personalize_rejects_invalid_request(
    personalizer: ContextAwarePersonalizer,
) -> None:
    with pytest.raises(TypeError):
        personalizer.personalize(
            "invalid",  # type: ignore[arg-type]
            PersonalizationResult(),
        )

def test_personalize_rejects_invalid_result(
    personalizer: ContextAwarePersonalizer,
) -> None:
    with pytest.raises(TypeError):
        personalizer.personalize(
            PersonalizationRequest(user_message="Hello."),
            "invalid",  # type: ignore[arg-type]
        )