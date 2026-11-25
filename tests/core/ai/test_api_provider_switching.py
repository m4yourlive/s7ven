
"""Integration tests for switching between API providers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.ai.context_manager import ConversationContext
from seven.core.ai.engine import AIEngine
from seven.core.ai.models import MessageRole
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig


def make_completion(content: str, model: str = "test-model"):
    """Build a mock OpenAI-compatible completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


@pytest.fixture
def multi_provider_pipeline():
    """Build one pipeline with two registered API providers."""
    openai_provider = OpenAIProvider(
        api_key="test-openai-key",
        model="gpt-4o-mini",
    )

    openrouter_provider = OpenAICompatibleProvider(
        api_key="test-openrouter-key",
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="test-model",
    )

    registry = ProviderRegistry()

    engine = AIEngine(
        provider=openai_provider,
        provider_registry=registry,
    )

    registry.register(openrouter_provider)

    config = AIConfig()

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=PromptManager(DEFAULT_PROMPTS),
    )

    context = ConversationContext()

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
        context=context,
    )

    return {
        "pipeline": pipeline,
        "engine": engine,
        "registry": registry,
        "openai": openai_provider,
        "openrouter": openrouter_provider,
        "context": context,
    }


@pytest.mark.asyncio
async def test_pipeline_switches_from_openai_to_openrouter(
    multi_provider_pipeline,
) -> None:
    components = multi_provider_pipeline

    pipeline = components["pipeline"]
    engine = components["engine"]
    openai = components["openai"]
    openrouter = components["openrouter"]

    openai._client.chat.completions.create = AsyncMock(
        return_value=make_completion(
            "First response from OpenAI.",
            "gpt-4o-mini",
        )
    )

    openrouter._client.chat.completions.create = AsyncMock(
        return_value=make_completion(
            "Second response from OpenRouter.",
            "test-model",
        )
    )

    original_context = pipeline.context

    first_response = await pipeline.generate("First message.")

    assert first_response.provider == "openai"
    assert first_response.content == "First response from OpenAI."

    selected_provider = await engine.switch_provider("openrouter")

    assert selected_provider is openrouter
    assert engine.provider is openrouter

    second_response = await pipeline.generate("Second message.")

    assert second_response.provider == "openrouter"
    assert second_response.content == "Second response from OpenRouter."

    second_request = (
        openrouter._client.chat.completions.create.call_args.kwargs
    )

    second_messages = second_request["messages"]

    assert any(
        message["role"] == MessageRole.USER.value
        and message["content"] == "First message."
        for message in second_messages
    )

    assert any(
        message["role"] == MessageRole.ASSISTANT.value
        and message["content"] == "First response from OpenAI."
        for message in second_messages
    )

    assert second_messages[-1]["role"] == MessageRole.USER.value
    assert second_messages[-1]["content"] == "Second message."

    assert pipeline.context is original_context
    assert pipeline.context.message_count == 4


@pytest.mark.asyncio
async def test_pipeline_switches_back_to_openai(
    multi_provider_pipeline,
) -> None:
    components = multi_provider_pipeline

    pipeline = components["pipeline"]
    engine = components["engine"]
    openai = components["openai"]
    openrouter = components["openrouter"]

    openai._client.chat.completions.create = AsyncMock(
        side_effect=[
            make_completion("OpenAI response one."),
            make_completion("OpenAI response two."),
        ]
    )

    openrouter._client.chat.completions.create = AsyncMock(
        return_value=make_completion("OpenRouter response."),
    )

    first_response = await pipeline.generate("First message.")

    assert first_response.provider == "openai"

    await engine.switch_provider("openrouter")

    second_response = await pipeline.generate("Second message.")

    assert second_response.provider == "openrouter"

    await engine.switch_provider("openai")

    assert engine.provider is openai

    third_response = await pipeline.generate("Third message.")

    assert third_response.provider == "openai"
    assert third_response.content == "OpenAI response two."

    third_request = (
        openai._client.chat.completions.create.call_args_list[-1].kwargs
    )

    third_messages = third_request["messages"]

    assert any(
        message["role"] == MessageRole.USER.value
        and message["content"] == "Second message."
        for message in third_messages
    )

    assert any(
        message["role"] == MessageRole.ASSISTANT.value
        and message["content"] == "OpenRouter response."
        for message in third_messages
    )

    assert third_messages[-1]["role"] == MessageRole.USER.value
    assert third_messages[-1]["content"] == "Third message."

    assert pipeline.context.message_count == 6


@pytest.mark.asyncio
async def test_registry_contains_all_api_providers(
    multi_provider_pipeline,
) -> None:
    components = multi_provider_pipeline
    registry = components["registry"]

    assert registry.has_provider("openai")
    assert registry.has_provider("openrouter")

    assert registry.get("openai") is components["openai"]
    assert registry.get("openrouter") is components["openrouter"]

    assert registry.list_providers() == (
        "openai",
        "openrouter",
    )


@pytest.mark.asyncio
async def test_pipeline_stream_uses_switched_api_provider(
    multi_provider_pipeline,
) -> None:
    components = multi_provider_pipeline

    pipeline = components["pipeline"]
    engine = components["engine"]
    openrouter = components["openrouter"]

    await engine.switch_provider("openrouter")

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="OpenRouter ")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="stream response.")
                )
            ]
        ),
    ]

    mock_stream = MagicMock()
    mock_stream.__aiter__.return_value = iter(chunks)
    mock_stream.close = AsyncMock()

    openrouter._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    result = [
        chunk
        async for chunk in pipeline.stream("Stream this response.")
    ]

    assert result == [
        "OpenRouter ",
        "stream response.",
    ]

    openrouter._client.chat.completions.create.assert_awaited_once()
    mock_stream.close.assert_awaited_once()

    assert engine.provider is openrouter
    assert pipeline.context.message_count == 2

    history = pipeline.context.history

    assert history[0].role is MessageRole.USER
    assert history[0].content == "Stream this response."

    assert history[1].role is MessageRole.ASSISTANT
    assert history[1].content == "OpenRouter stream response."


@pytest.mark.asyncio
async def test_switching_provider_does_not_close_previous_provider(
    multi_provider_pipeline,
) -> None:
    components = multi_provider_pipeline

    engine = components["engine"]
    openai = components["openai"]
    openrouter = components["openrouter"]

    openai._client.close = AsyncMock()
    openrouter._client.close = AsyncMock()

    await engine.switch_provider("openrouter")

    openai._client.close.assert_not_awaited()
    openrouter._client.close.assert_not_awaited()

    await engine.switch_provider("openai")

    openai._client.close.assert_not_awaited()
    openrouter._client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_closes_all_activated_api_providers(
    multi_provider_pipeline,
) -> None:
    components = multi_provider_pipeline

    engine = components["engine"]
    openai = components["openai"]
    openrouter = components["openrouter"]

    openai._client.close = AsyncMock()
    openrouter._client.close = AsyncMock()

    await engine.switch_provider("openrouter")
    await engine.switch_provider("openai")

    await engine.close()

    assert engine.is_closed is True

    openai._client.close.assert_awaited_once()
    openrouter._client.close.assert_awaited_once()
