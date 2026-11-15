
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.ai.exceptions import (
    AIProviderUnavailableError,
    AIResponseError,
)
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider


@pytest.fixture
def provider():
    return OpenAIProvider(
        api_key="test-api-key",
        model="gpt-4o-mini",
    )


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


def test_provider_name(provider):
    assert provider.name == "openai"


@pytest.mark.asyncio
async def test_generate_returns_response(provider, generation_request):
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello!"),
                finish_reason="stop",
            )
        ],
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
        ),
    )

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    response = await provider.generate(generation_request)

    assert response.content == "Hello!"
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.usage["total_tokens"] == 8
    assert response.usage["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_generate_empty_choices(provider, generation_request):
    provider._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[],
            model="gpt-4o-mini",
            usage=None,
        )
    )

    with pytest.raises(AIResponseError):
        await provider.generate(generation_request)


@pytest.mark.asyncio
async def test_stream_returns_chunks(provider, generation_request):
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello ")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Seven!")
                )
            ]
        ),
    ]

    mock_stream = MagicMock()
    mock_stream.__aiter__.return_value = iter(chunks)
    mock_stream.close = AsyncMock()

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream
    )

    result = [
        chunk
        async for chunk in provider.stream(generation_request)
    ]

    assert result == ["Hello ", "Seven!"]
    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_closed_provider_rejects_generation(
    provider,
    generation_request,
):
    await provider.close()

    with pytest.raises(AIProviderUnavailableError):
        await provider.generate(generation_request)


@pytest.mark.asyncio
async def test_closed_provider_rejects_stream(
    provider,
    generation_request,
):
    await provider.close()

    with pytest.raises(AIProviderUnavailableError):
        async for _ in provider.stream(generation_request):
            pass


@pytest.mark.asyncio
async def test_close_is_idempotent(provider):
    provider._client.close = AsyncMock()

    await provider.close()
    await provider.close()

    provider._client.close.assert_awaited_once()


def test_provider_rejects_empty_api_key():
    with pytest.raises(ValueError):
        OpenAIProvider(api_key=" ")


def test_provider_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        OpenAIProvider(
            api_key="test-api-key",
            timeout=0,
        )
