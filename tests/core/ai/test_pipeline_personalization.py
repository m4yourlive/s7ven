
"""Integration tests for AI pipeline personalization."""

from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.models import (
    GenerationResponse,
    MessageRole,
)
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


@pytest.fixture
def pipeline_components() -> tuple[AIResponsePipeline, OpenAIProvider]:
    """Build an AI pipeline with a test provider."""
    config = AIConfig()
    prompt_manager = PromptManager(DEFAULT_PROMPTS)

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=prompt_manager,
    )

    provider = OpenAIProvider(api_key="test-api-key")
    engine = AIEngine(provider)

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
    )

    return pipeline, provider


def _get_personalization_message(request):
    """Find the personalization system message in a generation request."""
    return next(
        (
            message
            for message in request.messages
            if (
                message.role is MessageRole.SYSTEM
                and (
                    "Response style instructions:" in message.content
                    or "Domain context instructions:" in message.content
                )
            )
        ),
        None,
    )


@pytest.mark.asyncio
async def test_generate_injects_personalization_context(
    pipeline_components: tuple[AIResponsePipeline, OpenAIProvider],
) -> None:
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="Python functions are reusable blocks of code.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

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
        ),
    )

    response = await pipeline.generate(
        "Explain Python functions and their implementation.",
        user_profile=profile,
    )

    assert response.content == (
        "Python functions are reusable blocks of code."
    )

    request = provider.generate.call_args.args[0]
    personalization_message = _get_personalization_message(request)

    assert personalization_message is not None
    assert "Keep communication concise" in personalization_message.content
    assert "fundamentals" in personalization_message.content

    assert request.messages[-1].role is MessageRole.USER
    assert request.messages[-1].content == (
        "Explain Python functions and their implementation."
    )


@pytest.mark.asyncio
async def test_generate_excludes_irrelevant_technical_personalization(
    pipeline_components: tuple[AIResponsePipeline, OpenAIProvider],
) -> None:
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="You could try pasta or vegetable soup.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    profile = UserProfile(
        preferences=(
            UserPreference(
                key="technical_depth",
                value="beginner",
            ),
        ),
    )

    await pipeline.generate(
        "Give me some dinner ideas.",
        user_profile=profile,
    )

    request = provider.generate.call_args.args[0]

    assert _get_personalization_message(request) is None
    assert request.messages[-1].role is MessageRole.USER


@pytest.mark.asyncio
async def test_generate_injects_relevant_domain_context(
    pipeline_components: tuple[AIResponsePipeline, OpenAIProvider],
) -> None:
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="Blockchain wallets use cryptographic keys.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    profile = UserProfile(
        preferences=(
            UserPreference(
                key="interests",
                value="blockchain technology",
            ),
        ),
    )

    await pipeline.generate(
        "Explain blockchain wallet security.",
        user_profile=profile,
    )

    request = provider.generate.call_args.args[0]
    personalization_message = _get_personalization_message(request)

    assert personalization_message is not None
    assert "Domain context instructions:" in personalization_message.content
    assert "blockchain technology" in personalization_message.content


@pytest.mark.asyncio
async def test_stream_injects_personalization_context(
    pipeline_components: tuple[AIResponsePipeline, OpenAIProvider],
) -> None:
    pipeline, provider = pipeline_components

    captured_request = None

    async def mock_stream(request):
        nonlocal captured_request
        captured_request = request

        yield "A concise "
        yield "Python explanation."

    provider.stream = mock_stream

    profile = UserProfile(
        preferences=(
            UserPreference(
                key="response_style",
                value="concise",
            ),
        ),
    )

    chunks = [
        chunk
        async for chunk in pipeline.stream(
            "What is Python?",
            user_profile=profile,
        )
    ]

    assert chunks == [
        "A concise ",
        "Python explanation.",
    ]

    assert captured_request is not None

    personalization_message = _get_personalization_message(
        captured_request
    )

    assert personalization_message is not None
    assert "Keep communication concise" in (
        personalization_message.content
    )

    assert pipeline.context.message_count == 2


@pytest.mark.asyncio
async def test_generate_rejects_invalid_user_profile(
    pipeline_components: tuple[AIResponsePipeline, OpenAIProvider],
) -> None:
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock()

    with pytest.raises(TypeError, match="user_profile"):
        await pipeline.generate(
            "Explain Python.",
            user_profile="invalid",  # type: ignore[arg-type]
        )

    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_stream_rejects_invalid_user_profile(
    pipeline_components: tuple[AIResponsePipeline, OpenAIProvider],
) -> None:
    pipeline, provider = pipeline_components

    with pytest.raises(TypeError, match="user_profile"):
        async for _ in pipeline.stream(
            "Explain Python.",
            user_profile="invalid",  # type: ignore[arg-type]
        ):
            pass

    assert pipeline.context.message_count == 0
