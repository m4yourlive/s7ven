
import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import (
    AIGenerationError,
    AIProviderNetworkError,
    AIResponseError,
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


@pytest.mark.asyncio
async def test_generate_normalizes_unexpected_exception(
    engine,
    provider,
    generation_request,
):
    original_error = RuntimeError("Unexpected runtime failure.")
    provider.generate = AsyncMock(side_effect=original_error)

    with pytest.raises(AIGenerationError) as exc_info:
        await engine.generate(generation_request)

    assert exc_info.value.__cause__ is original_error
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_generate_preserves_known_ai_error(
    engine,
    provider,
    generation_request,
):
    original_error = AIProviderNetworkError("Network unavailable.")
    provider.generate = AsyncMock(side_effect=original_error)

    with pytest.raises(AIProviderNetworkError) as exc_info:
        await engine.generate(generation_request)

    assert exc_info.value is original_error
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_generate_rejects_invalid_response(
    engine,
    provider,
    generation_request,
):
    provider.generate = AsyncMock(return_value="Invalid response")

    with pytest.raises(AIResponseError):
        await engine.generate(generation_request)

    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_generate_accepts_valid_response(
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
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_stream_normalizes_unexpected_exception(
    engine,
    provider,
    generation_request,
):
    original_error = RuntimeError("Unexpected stream failure.")

    async def failing_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        raise original_error
        yield ""

    provider.stream = failing_stream

    with pytest.raises(AIGenerationError) as exc_info:
        async for _ in engine.stream(generation_request):
            pass

    assert exc_info.value.__cause__ is original_error
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_stream_rejects_invalid_chunk(
    engine,
    provider,
    generation_request,
):
    async def invalid_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Valid chunk"
        yield 123

    provider.stream = invalid_stream

    chunks = []

    with pytest.raises(AIResponseError):
        async for chunk in engine.stream(generation_request):
            chunks.append(chunk)

    assert chunks == ["Valid chunk"]
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_stream_preserves_known_provider_error(
    engine,
    provider,
    generation_request,
):
    original_error = AIProviderNetworkError("Stream connection lost.")

    async def failing_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        raise original_error
        yield ""

    provider.stream = failing_stream

    with pytest.raises(AIProviderNetworkError) as exc_info:
        async for _ in engine.stream(generation_request):
            pass

    assert exc_info.value is original_error
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_generate_releases_operation_after_cancellation(
    engine,
    provider,
    generation_request,
):
    started = asyncio.Event()

    async def blocked_generation(
        request: GenerationRequest,
    ) -> GenerationResponse:
        started.set()
        await asyncio.Event().wait()

    provider.generate = blocked_generation

    task = asyncio.create_task(
        engine.generate(generation_request)
    )

    await started.wait()
    assert engine.active_operations == 1

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_engine_remains_available_after_generation_failure(
    engine,
    provider,
    generation_request,
):
    provider.generate = AsyncMock(
        side_effect=[
            RuntimeError("Temporary unexpected failure."),
            GenerationResponse(
                content="Recovered response",
                model="gpt-4o-mini",
                provider="openai",
            ),
        ]
    )

    with pytest.raises(AIGenerationError):
        await engine.generate(generation_request)

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="Recovered response",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    result = await engine.generate(generation_request)

    assert result.content == "Recovered response"
    assert engine.active_operations == 0
    assert engine.is_closed is False
