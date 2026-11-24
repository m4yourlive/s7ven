
"""Tests for API provider error handling and request compatibility."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.ai.exceptions import (
    AIAuthenticationError,
    AIConfigurationError,
    AIProviderCapabilityError,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRateLimitError,
    AIRequestError,
    AIResponseError,
)
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
    ProviderCapabilities,
)
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider


@pytest.fixture
def request_without_system() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Explain Python.",
            ),
        ),
        model="test-model",
        temperature=0.4,
        max_tokens=500,
        top_p=0.8,
        frequency_penalty=0.2,
        presence_penalty=0.1,
    )

@pytest.fixture
def request_with_system() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
            ),
            AIMessage(
                role=MessageRole.USER,
                content="Explain Python.",
            ),
        ),
        model="test-model",
    )

@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-api-key",
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="test-model",
    )

@pytest.mark.parametrize(
    ("status_code", "exception_type"),
    [
        (400, AIRequestError),
        (401, AIAuthenticationError),
        (403, AIAuthenticationError),
        (404, AIConfigurationError),
        (408, AIProviderTimeoutError),
        (422, AIRequestError),
        (429, AIRateLimitError),
        (500, AIProviderUnavailableError),
        (502, AIProviderUnavailableError),
        (503, AIProviderUnavailableError),
        (504, AIProviderTimeoutError),
    ],
)
def test_http_status_errors_are_classified(
    provider: OpenAICompatibleProvider,
    status_code: int,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        provider._raise_status_error(
            status_code,
            operation="request",
        )

def test_unknown_http_status_maps_to_provider_error(
    provider: OpenAICompatibleProvider,
) -> None:
    with pytest.raises(AIProviderError, match="HTTP 418"):
        provider._raise_status_error(
            418,
            operation="request",
        )

def test_capabilities_filter_unsupported_parameters(
    request_without_system: GenerationRequest,
) -> None:
    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_system_messages=True,
        supports_temperature=False,
        supports_max_tokens=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
    )

    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        provider_name="limited-api",
        base_url="https://example.com/v1",
        model="test-model",
        capabilities=capabilities,
    )

    options = provider._build_request_options(request_without_system)

    assert options["model"] == "test-model"
    assert "messages" in options
    assert "temperature" not in options
    assert "max_tokens" not in options
    assert "top_p" not in options
    assert "frequency_penalty" not in options
    assert "presence_penalty" not in options

def test_capabilities_preserve_supported_parameters(
    request_without_system: GenerationRequest,
) -> None:
    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        provider_name="limited-api",
        base_url="https://example.com/v1",
        model="test-model",
        capabilities=ProviderCapabilities(
            supports_temperature=True,
            supports_max_tokens=True,
            supports_top_p=False,
            supports_frequency_penalty=True,
            supports_presence_penalty=False,
        ),
    )

    options = provider._build_request_options(request_without_system)

    assert options["temperature"] == 0.4
    assert options["max_tokens"] == 500
    assert "top_p" not in options
    assert options["frequency_penalty"] == 0.2
    assert "presence_penalty" not in options

def test_unsupported_system_messages_are_rejected(
    request_with_system: GenerationRequest,
) -> None:
    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        provider_name="limited-api",
        base_url="https://example.com/v1",
        model="test-model",
        capabilities=ProviderCapabilities(
            supports_system_messages=False,
        ),
    )

    with pytest.raises(
        AIProviderCapabilityError,
        match="system messages",
    ):
        provider._build_request_options(request_with_system)

@pytest.mark.asyncio
async def test_streaming_rejects_unsupported_capability(
    request_without_system: GenerationRequest,
) -> None:
    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        provider_name="limited-api",
        base_url="https://example.com/v1",
        model="test-model",
        capabilities=ProviderCapabilities(
            supports_streaming=False,
        ),
    )

    provider._client.chat.completions.create = AsyncMock()

    with pytest.raises(
        AIProviderCapabilityError,
        match="streaming",
    ):
        async for _ in provider.stream(request_without_system):
            pass

    provider._client.chat.completions.create.assert_not_awaited()

@pytest.mark.asyncio
async def test_generate_rejects_empty_response(
    provider: OpenAICompatibleProvider,
    request_without_system: GenerationRequest,
) -> None:
    provider._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[],
            model="test-model",
            usage=None,
        )
    )

    with pytest.raises(AIResponseError, match="empty response"):
        await provider.generate(request_without_system)

@pytest.mark.asyncio
async def test_generate_rejects_empty_message_content(
    provider: OpenAICompatibleProvider,
    request_without_system: GenerationRequest,
) -> None:
    provider._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  "),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=None,
        )
    )

    with pytest.raises(AIResponseError, match="empty or invalid message"):
        await provider.generate(request_without_system)

@pytest.mark.asyncio
async def test_generate_handles_missing_usage(
    provider: OpenAICompatibleProvider,
    request_without_system: GenerationRequest,
) -> None:
    provider._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Valid response."),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=None,
        )
    )

    response = await provider.generate(request_without_system)

    assert response.content == "Valid response."
    assert response.usage == {}

@pytest.mark.asyncio
async def test_generate_falls_back_to_request_model(
    provider: OpenAICompatibleProvider,
    request_without_system: GenerationRequest,
) -> None:
    provider._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Valid response."),
                    finish_reason="stop",
                )
            ],
            model=None,
            usage=None,
        )
    )

    response = await provider.generate(request_without_system)

    assert response.model == request_without_system.model

@pytest.mark.asyncio
async def test_stream_closes_after_cleanup_failure(
    provider: OpenAICompatibleProvider,
    request_without_system: GenerationRequest,
) -> None:
    mock_stream = MagicMock()
    mock_stream.__aiter__.return_value = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Valid chunk.")
                    )
                ]
            )
        ]
    )
    mock_stream.close = AsyncMock(
        side_effect=RuntimeError("Cleanup failed.")
    )

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream
    )

    with pytest.raises(
        AIProviderError,
        match="stream cleanup failed",
    ):
        async for _ in provider.stream(request_without_system):
            pass

    mock_stream.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_stream_preserves_original_iteration_error(
    provider: OpenAICompatibleProvider,
    request_without_system: GenerationRequest,
) -> None:
    mock_stream = MagicMock()

    async def failing_iterator():
        raise RuntimeError("Original stream failure.")
        yield "unreachable"

    mock_stream.__aiter__.side_effect = failing_iterator
    mock_stream.close = AsyncMock(
        side_effect=RuntimeError("Cleanup failed.")
    )

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream
    )

    with pytest.raises(RuntimeError, match="Original stream failure"):
        async for _ in provider.stream(request_without_system):
            pass

    mock_stream.close.assert_awaited_once()

def test_native_openai_provider_enables_function_calling_by_default() -> None:
    provider = OpenAIProvider(api_key="test-api-key")

    assert provider.capabilities.supports_function_calling is True