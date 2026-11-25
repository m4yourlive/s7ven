
"""Tests for the local AI provider adapter."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.exceptions import AIProviderUnavailableError
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)
from seven.core.ai.providers.local_ai_provider import LocalAIProvider


class FakeLocalRuntime(LocalModelRuntime):
    """Test runtime for the local AI provider."""

    def __init__(self) -> None:
        self.generate_mock = AsyncMock(
            return_value=GenerationResponse(
                content="Local response",
                model="test-model",
                provider="test-runtime",
            )
        )
        self.close_mock = AsyncMock()
        self.capabilities_value = ProviderCapabilities(
            supports_streaming=True,
            supports_system_messages=True,
            supports_temperature=True,
            supports_max_tokens=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        )

    @property
    def runtime_name(self) -> str:
        return "test-runtime"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.capabilities_value

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return await self.generate_mock(request)

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Local "
        yield "response"

    async def close(self) -> None:
        await self.close_mock()


@pytest.fixture
def runtime() -> FakeLocalRuntime:
    return FakeLocalRuntime()


@pytest.fixture
def provider(runtime: FakeLocalRuntime) -> LocalAIProvider:
    return LocalAIProvider(runtime)


def test_provider_implements_ai_provider(
    provider: LocalAIProvider,
) -> None:
    assert isinstance(provider, AIProvider)


def test_provider_has_default_name(
    provider: LocalAIProvider,
) -> None:
    assert provider.name == "local"


def test_provider_normalizes_custom_name(
    runtime: FakeLocalRuntime,
) -> None:
    provider = LocalAIProvider(runtime, provider_name="  LOCAL-OLLAMA ")

    assert provider.name == "local-ollama"


def test_provider_rejects_empty_name(
    runtime: FakeLocalRuntime,
) -> None:
    with pytest.raises(ValueError, match="provider_name"):
        LocalAIProvider(runtime, provider_name="  ")


def test_provider_rejects_invalid_runtime() -> None:
    with pytest.raises(TypeError, match="LocalModelRuntime"):
        LocalAIProvider(runtime=object())  # type: ignore[arg-type]


def test_provider_exposes_runtime(
    provider: LocalAIProvider,
    runtime: FakeLocalRuntime,
) -> None:
    assert provider.runtime is runtime


def test_provider_exposes_runtime_capabilities(
    provider: LocalAIProvider,
    runtime: FakeLocalRuntime,
) -> None:
    assert provider.capabilities is runtime.capabilities


@pytest.mark.asyncio
async def test_generate_delegates_to_runtime(
    provider: LocalAIProvider,
    runtime: FakeLocalRuntime,
) -> None:
    request = AsyncMock(spec=GenerationRequest)

    response = await provider.generate(request)

    runtime.generate_mock.assert_awaited_once_with(request)
    assert response.content == "Local response"
    assert response.provider == "test-runtime"


@pytest.mark.asyncio
async def test_stream_delegates_to_runtime(
    provider: LocalAIProvider,
) -> None:
    request = AsyncMock(spec=GenerationRequest)

    chunks = [
        chunk
        async for chunk in provider.stream(request)
    ]

    assert chunks == ["Local ", "response"]


@pytest.mark.asyncio
async def test_close_closes_runtime_once(
    provider: LocalAIProvider,
    runtime: FakeLocalRuntime,
) -> None:
    await provider.close()
    await provider.close()

    runtime.close_mock.assert_awaited_once()
    assert provider.is_closed is True


@pytest.mark.asyncio
async def test_generate_rejects_closed_provider(
    provider: LocalAIProvider,
) -> None:
    await provider.close()

    with pytest.raises(AIProviderUnavailableError, match="closed"):
        await provider.generate(AsyncMock(spec=GenerationRequest))


@pytest.mark.asyncio
async def test_stream_rejects_closed_provider(
    provider: LocalAIProvider,
) -> None:
    await provider.close()

    with pytest.raises(AIProviderUnavailableError, match="closed"):
        async for _ in provider.stream(AsyncMock(spec=GenerationRequest)):
            pass
