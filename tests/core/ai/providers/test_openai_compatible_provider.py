
"""Tests for OpenAI-compatible API providers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)


@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-api-key",
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="test-model",
    )


@pytest.fixture
def generation_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Hello Seven.",
            ),
        ),
        model="test-model",
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.1,
    )


def test_provider_name_is_normalized(
    provider: OpenAICompatibleProvider,
) -> None:
    assert provider.name == "openrouter"


def test_provider_uses_configured_model(
    provider: OpenAICompatibleProvider,
) -> None:
    assert provider._model == "test-model"


def test_provider_uses_custom_base_url(
    provider: OpenAICompatibleProvider,
) -> None:
    assert str(provider._client.base_url).rstrip("/") == (
        "https://openrouter.ai/api/v1"
    )


def test_provider_builds_compatible_request_options(
    provider: OpenAICompatibleProvider,
    generation_request: GenerationRequest,
) -> None:
    options = provider._build_request_options(generation_request)

    assert options["model"] == "test-model"
    assert options["temperature"] == 0.5
    assert options["top_p"] == 0.9
    assert options["frequency_penalty"] == 0.2
    assert options["presence_penalty"] == 0.1

    assert options["messages"] == [
        {
            "role": "user",
            "content": "Hello Seven.",
        }
    ]


@pytest.mark.asyncio
async def test_generate_returns_configured_provider_name(
    provider: OpenAICompatibleProvider,
    generation_request: GenerationRequest,
) -> None:
    provider._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello from OpenRouter."),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=4,
                total_tokens=9,
            ),
        )
    )

    response = await provider.generate(generation_request)

    assert response.content == "Hello from OpenRouter."
    assert response.provider == "openrouter"
    assert response.model == "test-model"
    assert response.usage["total_tokens"] == 9

    provider._client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_returns_compatible_provider_chunks(
    provider: OpenAICompatibleProvider,
    generation_request: GenerationRequest,
) -> None:
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
                    delta=SimpleNamespace(content="from provider.")
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

    assert result == ["Hello ", "from provider."]
    mock_stream.close.assert_awaited_once()


def test_provider_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="Provider name"):
        OpenAICompatibleProvider(
            api_key="test-api-key",
            provider_name=" ",
            base_url="https://example.com/v1",
            model="test-model",
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        " ",
        "example.com/v1",
        "ftp://example.com/v1",
        "https://",
    ],
)
def test_provider_rejects_invalid_base_url(
    base_url: str,
) -> None:
    with pytest.raises(ValueError, match="Base URL"):
        OpenAICompatibleProvider(
            api_key="test-api-key",
            provider_name="custom",
            base_url=base_url,
            model="test-model",
        )


def test_provider_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="Model name"):
        OpenAICompatibleProvider(
            api_key="test-api-key",
            provider_name="custom",
            base_url="https://example.com/v1",
            model=" ",
        )


def test_provider_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleProvider(
            api_key=" ",
            provider_name="custom",
            base_url="https://example.com/v1",
            model="test-model",
        )


def test_provider_rejects_openai_provider_name() -> None:
    with pytest.raises(ValueError, match="OpenAIProvider"):
        OpenAICompatibleProvider(
            api_key="test-api-key",
            provider_name="openai",
            base_url="https://example.com/v1",
            model="test-model",
        )
