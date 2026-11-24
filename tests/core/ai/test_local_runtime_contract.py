
"""Tests for the local AI runtime contract."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)


class FakeLocalRuntime(LocalModelRuntime):
    """Test implementation of the local runtime contract."""

    @property
    def runtime_name(self) -> str:
        return "fake-local"

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return GenerationResponse(
            content="Local response",
            model="test-model",
            provider="fake-local",
        )

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Local "
        yield "response"

    async def close(self) -> None:
        return None


def test_local_runtime_is_abstract() -> None:
    with pytest.raises(TypeError):
        LocalModelRuntime()


def test_local_runtime_requires_runtime_name() -> None:
    class MissingRuntimeName(LocalModelRuntime):
        async def generate(
            self,
            request: GenerationRequest,
        ) -> GenerationResponse:
            raise NotImplementedError

        async def stream(
            self,
            request: GenerationRequest,
        ) -> AsyncIterator[str]:
            yield ""

        async def close(self) -> None:
            return None

    with pytest.raises(TypeError):
        MissingRuntimeName()


def test_local_runtime_exposes_runtime_name() -> None:
    runtime = FakeLocalRuntime()

    assert runtime.runtime_name == "fake-local"


def test_local_runtime_has_default_capabilities() -> None:
    runtime = FakeLocalRuntime()

    assert isinstance(runtime.capabilities, ProviderCapabilities)
    assert runtime.capabilities.supports_streaming is True
    assert runtime.capabilities.supports_system_messages is True


def test_local_runtime_can_override_capabilities() -> None:
    class LimitedLocalRuntime(FakeLocalRuntime):
        @property
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(
                supports_streaming=False,
                supports_system_messages=True,
                supports_temperature=False,
                supports_max_tokens=True,
                supports_top_p=False,
                supports_frequency_penalty=False,
                supports_presence_penalty=False,
            )

    runtime = LimitedLocalRuntime()

    assert runtime.capabilities.supports_streaming is False
    assert runtime.capabilities.supports_temperature is False


@pytest.mark.asyncio
async def test_local_runtime_generates_response() -> None:
    runtime = FakeLocalRuntime()
    request = AsyncMock(spec=GenerationRequest)

    response = await runtime.generate(request)

    assert isinstance(response, GenerationResponse)
    assert response.content == "Local response"
    assert response.model == "test-model"
    assert response.provider == "fake-local"


@pytest.mark.asyncio
async def test_local_runtime_streams_response() -> None:
    runtime = FakeLocalRuntime()
    request = AsyncMock(spec=GenerationRequest)

    chunks = [
        chunk
        async for chunk in runtime.stream(request)
    ]

    assert chunks == ["Local ", "response"]


@pytest.mark.asyncio
async def test_local_runtime_closes() -> None:
    runtime = FakeLocalRuntime()

    await runtime.close()
