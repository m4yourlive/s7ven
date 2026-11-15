from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import (
    AIProviderUnavailableError,
    AIRequestError,
)
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    GenerationResponse,
    MessageRole,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider


@pytest.fixture
def provider():
    return OpenAIProvider(api_key="test-api-key")


@pytest.fixture
def engine(provider):
    return AIEngine(provider)


@pytest.fixture
def generation_request():
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Hello Seven",
            ),
        ),
        model="gpt-4o-mini",
    )


def test_engine_initialization(provider):
    engine = AIEngine(provider)

    assert engine.provider is provider
    assert engine.is_closed is False


def test_engine_rejects_invalid_provider():
    with pytest.raises(TypeError):
        AIEngine(object())


@pytest.mark.asyncio
async def test_generate_delegates_to_provider(
    engine,
    provider,
    generation_request,
):
    expected = GenerationResponse(
        content="Hello!",
        model="gpt-4o-mini",
        provider="openai",
    )
    provider.generate = AsyncMock(return_value=expected)

    result = await engine.generate(generation_request)

    assert result is expected
    provider.generate.assert_awaited_once_with(generation_request)


@pytest.mark.asyncio
async def test_generate_rejects_invalid_request(engine):
    with pytest.raises(AIRequestError):
        await engine.generate("invalid request")


@pytest.mark.asyncio
async def test_stream_delegates_to_provider(
    engine,
    provider,
    generation_request,
):
    async def mock_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Hello "
        yield ""
        yield "Seven!"

    provider.stream = mock_stream

    chunks = [
        chunk
        async for chunk in engine.stream(generation_request)
    ]

    assert chunks == ["Hello ", "Seven!"]


@pytest.mark.asyncio
async def test_stream_rejects_invalid_request(engine):
    with pytest.raises(AIRequestError):
        async for _ in engine.stream("invalid request"):
            pass


@pytest.mark.asyncio
async def test_generate_rejects_closed_engine(
    engine,
    generation_request,
):
    await engine.close()

    with pytest.raises(AIProviderUnavailableError):
        await engine.generate(generation_request)


@pytest.mark.asyncio
async def test_stream_rejects_closed_engine(
    engine,
    generation_request,
):
    await engine.close()

    with pytest.raises(AIProviderUnavailableError):
        async for _ in engine.stream(generation_request):
            pass


@pytest.mark.asyncio
async def test_close_closes_provider(provider):
    provider.close = AsyncMock()
    engine = AIEngine(provider)

    await engine.close()

    assert engine.is_closed is True
    provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_is_idempotent(provider):
    provider.close = AsyncMock()
    engine = AIEngine(provider)

    await engine.close()
    await engine.close()

    provider.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_error_is_propagated(
    engine,
    provider,
    generation_request,
):
    error = AIProviderUnavailableError("Provider unavailable")
    provider.generate = AsyncMock(side_effect=error)

    with pytest.raises(AIProviderUnavailableError):
        await engine.generate(generation_request)
