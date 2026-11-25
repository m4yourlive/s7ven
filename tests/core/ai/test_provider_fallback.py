
"""Tests for AI provider fallback behavior."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import (
    AIAuthenticationError,
    AIProviderNetworkError,
)
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
)
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.retry_policy import RetryPolicy


class NamedOpenAIProvider(OpenAIProvider):
    """OpenAI provider with a configurable test identifier."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(api_key="test-api-key")
        self._test_provider_name = provider_name

    @property
    def name(self) -> str:
        return self._test_provider_name


@pytest.fixture
def primary_provider():
    return NamedOpenAIProvider("primary")


@pytest.fixture
def fallback_provider():
    return NamedOpenAIProvider("fallback")


@pytest.fixture
def provider_registry(primary_provider, fallback_provider):
    registry = ProviderRegistry()
    registry.register(primary_provider)
    registry.register(fallback_provider)
    return registry


@pytest.fixture
def retry_policy():
    return RetryPolicy(
        max_attempts=1,
        initial_delay=0,
        max_delay=0,
    )


@pytest.fixture
def generation_request():
    from seven.core.ai.models import AIMessage, MessageRole

    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Hello Seven",
            ),
        ),
        model="gpt-4o-mini",
    )


@pytest.fixture
def expected_response():
    return GenerationResponse(
        content="Fallback response",
        model="gpt-4o-mini",
        provider="fallback",
    )


@pytest.mark.asyncio
async def test_generate_uses_fallback_provider(
    primary_provider,
    fallback_provider,
    provider_registry,
    retry_policy,
    generation_request,
    expected_response,
):
    primary_provider.generate = AsyncMock(
        side_effect=AIProviderNetworkError("Primary unavailable.")
    )
    fallback_provider.generate = AsyncMock(
        return_value=expected_response
    )

    engine = AIEngine(
        primary_provider,
        provider_registry,
        retry_policy,
    )

    result = await engine.generate(generation_request)

    assert result is expected_response
    primary_provider.generate.assert_awaited_once_with(generation_request)
    fallback_provider.generate.assert_awaited_once_with(generation_request)

    assert engine.provider is primary_provider
    assert engine.provider_name == "primary"


@pytest.mark.asyncio
async def test_generate_retries_primary_before_fallback(
    primary_provider,
    fallback_provider,
    provider_registry,
    generation_request,
    expected_response,
):
    primary_provider.generate = AsyncMock(
        side_effect=[
            AIProviderNetworkError("Temporary failure."),
            AIProviderNetworkError("Still unavailable."),
        ]
    )
    fallback_provider.generate = AsyncMock(
        return_value=expected_response
    )

    policy = RetryPolicy(
        max_attempts=2,
        initial_delay=0,
        max_delay=0,
    )
    engine = AIEngine(primary_provider, provider_registry, policy)

    result = await engine.generate(generation_request)

    assert result is expected_response
    assert primary_provider.generate.await_count == 2
    fallback_provider.generate.assert_awaited_once_with(generation_request)


@pytest.mark.asyncio
async def test_generate_does_not_fallback_on_permanent_error(
    primary_provider,
    fallback_provider,
    provider_registry,
    retry_policy,
    generation_request,
):
    error = AIAuthenticationError("Invalid API credentials.")
    primary_provider.generate = AsyncMock(side_effect=error)
    fallback_provider.generate = AsyncMock()

    engine = AIEngine(
        primary_provider,
        provider_registry,
        retry_policy,
    )

    with pytest.raises(AIAuthenticationError):
        await engine.generate(generation_request)

    fallback_provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_fallback_provider_is_managed(
    primary_provider,
    fallback_provider,
    provider_registry,
    retry_policy,
    generation_request,
    expected_response,
):
    primary_provider.generate = AsyncMock(
        side_effect=AIProviderNetworkError("Primary unavailable.")
    )
    fallback_provider.generate = AsyncMock(
        return_value=expected_response
    )
    primary_provider.close = AsyncMock()
    fallback_provider.close = AsyncMock()

    engine = AIEngine(
        primary_provider,
        provider_registry,
        retry_policy,
    )

    await engine.generate(generation_request)
    await engine.close()

    primary_provider.close.assert_awaited_once()
    fallback_provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_uses_fallback_before_first_chunk(
    primary_provider,
    fallback_provider,
    provider_registry,
    retry_policy,
    generation_request,
):
    async def failing_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        raise AIProviderNetworkError("Primary stream unavailable.")
        yield ""

    async def fallback_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Fallback "
        yield "response"

    primary_provider.stream = failing_stream
    fallback_provider.stream = fallback_stream

    engine = AIEngine(
        primary_provider,
        provider_registry,
        retry_policy,
    )

    chunks = [
        chunk
        async for chunk in engine.stream(generation_request)
    ]

    assert chunks == ["Fallback ", "response"]
    assert engine.provider is primary_provider


@pytest.mark.asyncio
async def test_stream_does_not_fallback_after_emitting_chunk(
    primary_provider,
    fallback_provider,
    provider_registry,
    retry_policy,
    generation_request,
):
    fallback_called = False

    async def failing_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Partial response"
        raise AIProviderNetworkError("Connection lost.")

    async def fallback_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        nonlocal fallback_called
        fallback_called = True
        yield "Fallback response"

    primary_provider.stream = failing_stream
    fallback_provider.stream = fallback_stream

    engine = AIEngine(
        primary_provider,
        provider_registry,
        retry_policy,
    )

    chunks = []

    with pytest.raises(AIProviderNetworkError):
        async for chunk in engine.stream(generation_request):
            chunks.append(chunk)

    assert chunks == ["Partial response"]
    assert fallback_called is False


@pytest.mark.asyncio
async def test_stream_retries_before_using_fallback(
    primary_provider,
    fallback_provider,
    provider_registry,
    generation_request,
):
    primary_calls = 0

    async def failing_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        nonlocal primary_calls
        primary_calls += 1
        raise AIProviderNetworkError("Primary unavailable.")
        yield ""

    async def fallback_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Fallback response"

    primary_provider.stream = failing_stream
    fallback_provider.stream = fallback_stream

    policy = RetryPolicy(
        max_attempts=2,
        initial_delay=0,
        max_delay=0,
    )
    engine = AIEngine(primary_provider, provider_registry, policy)

    chunks = [
        chunk
        async for chunk in engine.stream(generation_request)
    ]

    assert primary_calls == 2
    assert chunks == ["Fallback response"]


@pytest.mark.asyncio
async def test_fallback_is_unavailable_without_registry(
    primary_provider,
    fallback_provider,
    retry_policy,
    generation_request,
):
    primary_provider.generate = AsyncMock(
        side_effect=AIProviderNetworkError("Primary unavailable.")
    )
    fallback_provider.generate = AsyncMock()

    engine = AIEngine(primary_provider, retry_policy=retry_policy)

    with pytest.raises(AIProviderNetworkError):
        await engine.generate(generation_request)

    fallback_provider.generate.assert_not_awaited()
