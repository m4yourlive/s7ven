
"""Integration tests for provider-agnostic AI runtime validation."""

from collections.abc import AsyncIterator

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.interfaces.provider import AIProvider
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    MessageRole,
    ProviderCapabilities,
)
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers.local_ai_provider import LocalAIProvider
from seven.core.ai.providers.registry import ProviderRegistry
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig


class FakeAPIProvider(AIProvider):
    """API provider implementation for integration testing."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "api"

    async def generate(
        self,
        generation_request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(generation_request)

        return GenerationResponse(
            content="API response",
            model=generation_request.model,
            provider=self.name,
        )

    async def stream(
        self,
        generation_request: GenerationRequest,
    ) -> AsyncIterator[str]:
        self.requests.append(generation_request)
        yield "API "
        yield "stream response"

    async def close(self) -> None:
        self.closed = True


class FakeLocalRuntime(LocalModelRuntime):
    """Local runtime implementation for integration testing."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.closed = False

    @property
    def runtime_name(self) -> str:
        return "fake-local"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_system_messages=True,
            supports_temperature=True,
            supports_max_tokens=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        )

    async def generate(
        self,
        generation_request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(generation_request)

        return GenerationResponse(
            content="Local response",
            model=generation_request.model,
            provider=self.runtime_name,
        )

    async def stream(
        self,
        generation_request: GenerationRequest,
    ) -> AsyncIterator[str]:
        self.requests.append(generation_request)
        yield "Local "
        yield "stream response"

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def prompt_manager() -> PromptManager:
    return PromptManager(DEFAULT_PROMPTS)


def create_pipeline(
    config: AIConfig,
    provider: AIProvider,
) -> tuple[AIResponsePipeline, AIEngine]:
    """Create a pipeline using the shared provider contract."""
    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=PromptManager(DEFAULT_PROMPTS),
    )

    registry = ProviderRegistry()
    engine = AIEngine(
        provider=provider,
        provider_registry=registry,
    )

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
    )

    return pipeline, engine


@pytest.mark.asyncio
async def test_api_provider_uses_shared_pipeline_contract() -> None:
    config = AIConfig(
        mode="api",
        provider="openai",
        model="gpt-4o-mini",
    )
    provider = FakeAPIProvider()

    pipeline, engine = create_pipeline(config, provider)

    response = await pipeline.generate("Hello from API.")

    assert response.content == "API response"
    assert response.provider == "api"
    assert response.model == "gpt-4o-mini"

    assert len(provider.requests) == 1
    assert provider.requests[0].messages[-1].role is MessageRole.USER
    assert provider.requests[0].messages[-1].content == "Hello from API."

    await engine.close()
    assert provider.closed is True


@pytest.mark.asyncio
async def test_local_provider_uses_shared_pipeline_contract() -> None:
    config = AIConfig(
        mode="local",
        local_runtime="fake-local",
        local_model="local-test-model",
    )
    runtime = FakeLocalRuntime()
    provider = LocalAIProvider(
        runtime=runtime,
        provider_name="local",
    )

    pipeline, engine = create_pipeline(config, provider)

    response = await pipeline.generate("Hello from Local AI.")

    assert response.content == "Local response"
    assert response.model == "local-test-model"

    assert len(runtime.requests) == 1
    assert runtime.requests[0].messages[-1].role is MessageRole.USER
    assert runtime.requests[0].messages[-1].content == (
        "Hello from Local AI."
    )

    await engine.close()
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_api_provider_supports_shared_streaming_pipeline() -> None:
    config = AIConfig(
        mode="api",
        provider="openai",
        model="gpt-4o-mini",
    )
    provider = FakeAPIProvider()

    pipeline, engine = create_pipeline(config, provider)

    chunks = [
        chunk
        async for chunk in pipeline.stream("Stream from API.")
    ]

    assert chunks == ["API ", "stream response"]
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "gpt-4o-mini"

    assert pipeline.context.message_count == 2

    await engine.close()


@pytest.mark.asyncio
async def test_local_provider_supports_shared_streaming_pipeline() -> None:
    config = AIConfig(
        mode="local",
        local_runtime="fake-local",
        local_model="local-test-model",
    )
    runtime = FakeLocalRuntime()
    provider = LocalAIProvider(
        runtime=runtime,
        provider_name="local",
    )

    pipeline, engine = create_pipeline(config, provider)

    chunks = [
        chunk
        async for chunk in pipeline.stream("Stream from Local AI.")
    ]

    assert chunks == ["Local ", "stream response"]
    assert len(runtime.requests) == 1
    assert runtime.requests[0].model == "local-test-model"

    assert pipeline.context.message_count == 2

    await engine.close()


@pytest.mark.asyncio
async def test_local_provider_preserves_capability_contract() -> None:
    runtime = FakeLocalRuntime()
    provider = LocalAIProvider(
        runtime=runtime,
        provider_name="local",
    )

    capabilities = provider.capabilities

    assert capabilities.supports_streaming is True
    assert capabilities.supports_system_messages is True
    assert capabilities.supports_temperature is True
    assert capabilities.supports_max_tokens is True
    assert capabilities.supports_top_p is True
    assert capabilities.supports_frequency_penalty is False
    assert capabilities.supports_presence_penalty is False

    await provider.close()
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_provider_switch_keeps_pipeline_instance_and_context() -> None:
    config = AIConfig(
        mode="api",
        provider="openai",
        model="gpt-4o-mini",
    )

    api_provider = FakeAPIProvider()
    local_runtime = FakeLocalRuntime()
    local_provider = LocalAIProvider(
        runtime=local_runtime,
        provider_name="local",
    )

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=PromptManager(DEFAULT_PROMPTS),
    )

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        provider=api_provider,
        provider_registry=registry,
    )

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
    )

    original_pipeline = pipeline

    await pipeline.generate("First message.")

    await engine.switch_provider("local")

    assert pipeline is original_pipeline

    await pipeline.generate("Second message.")

    assert pipeline.context.message_count == 4
    assert len(api_provider.requests) == 1
    assert len(local_runtime.requests) == 1

    local_request = local_runtime.requests[0]

    assert any(
        message.role is MessageRole.ASSISTANT
        and message.content == "API response"
        for message in local_request.messages
    )

    await engine.close()
    assert api_provider.closed is True
    assert local_runtime.closed is True
