
"""Tests for AI engine provider switching and lifecycle management."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import (
    AIProviderUnavailableError,
)
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    GenerationResponse,
    MessageRole,
    ProviderCapabilities,
)
from seven.core.ai.providers import ProviderRegistry


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
    )


class DummyProvider(AIProvider):
    """Minimal provider for lifecycle tests."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.close_mock = AsyncMock()
        self.generate_mock = AsyncMock(
            side_effect=self._generate_response
        )
        self.generate_calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def _generate_response(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.generate_calls += 1

        return GenerationResponse(
            content=f"Response from {self.name}.",
            model=request.model,
            provider=self.name,
        )

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return await self.generate_mock(request)

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield f"Response from {self.name}."

    async def close(self) -> None:
        await self.close_mock()


class BlockingProvider(DummyProvider):
    """Provider that allows tests to control active operations."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.started.set()
        await self.release.wait()

        return GenerationResponse(
            content=f"Response from {self.name}.",
            model=request.model,
            provider=self.name,
        )


class ParallelProvider(DummyProvider):
    """Provider that records overlapping generation requests."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started_count = 0
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.started_count += 1

        if self.started_count >= 2:
            self.both_started.set()

        await self.release.wait()

        return GenerationResponse(
            content="Concurrent response.",
            model=request.model,
            provider=self.name,
        )


def test_engine_registers_initial_provider() -> None:
    provider = DummyProvider("openai")
    registry = ProviderRegistry()

    engine = AIEngine(
        provider,
        provider_registry=registry,
    )

    assert registry.get("openai") is provider
    assert engine.provider is provider
    assert engine.provider_name == "openai"


def test_engine_accepts_existing_provider_registration() -> None:
    provider = DummyProvider("openai")
    registry = ProviderRegistry()
    registry.register(provider)

    engine = AIEngine(
        provider,
        provider_registry=registry,
    )

    assert engine.provider_registry is registry
    assert engine.provider is provider


def test_engine_rejects_conflicting_provider_registration() -> None:
    registered_provider = DummyProvider("openai")
    active_provider = DummyProvider("openai")

    registry = ProviderRegistry()
    registry.register(registered_provider)

    with pytest.raises(ValueError, match="different instance"):
        AIEngine(
            active_provider,
            provider_registry=registry,
        )


def test_engine_rejects_invalid_registry() -> None:
    with pytest.raises(TypeError):
        AIEngine(
            DummyProvider("openai"),
            provider_registry=object(),  # type: ignore[arg-type]
        )


def test_engine_exposes_active_provider_capabilities() -> None:
    provider = DummyProvider("openai")
    engine = AIEngine(provider)

    assert engine.capabilities == ProviderCapabilities()
    assert engine.provider_name == "openai"


@pytest.mark.asyncio
async def test_switch_provider_changes_active_provider(
    generation_request: GenerationRequest,
) -> None:
    openai_provider = DummyProvider("openai")
    local_provider = DummyProvider("local")

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        openai_provider,
        provider_registry=registry,
    )

    selected = await engine.switch_provider("local")

    assert selected is local_provider
    assert engine.provider is local_provider
    assert engine.provider_name == "local"

    response = await engine.generate(generation_request)

    assert response.provider == "local"


@pytest.mark.asyncio
async def test_switch_provider_is_case_insensitive() -> None:
    initial_provider = DummyProvider("openai")
    local_provider = DummyProvider("local")

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        initial_provider,
        provider_registry=registry,
    )

    await engine.switch_provider("LOCAL")

    assert engine.provider is local_provider


@pytest.mark.asyncio
async def test_switch_provider_returns_current_provider() -> None:
    provider = DummyProvider("openai")
    registry = ProviderRegistry()

    engine = AIEngine(
        provider,
        provider_registry=registry,
    )

    selected = await engine.switch_provider("openai")

    assert selected is provider
    assert engine.provider is provider


@pytest.mark.asyncio
async def test_switch_provider_requires_registry() -> None:
    engine = AIEngine(DummyProvider("openai"))

    with pytest.raises(
        AIProviderUnavailableError,
        match="ProviderRegistry",
    ):
        await engine.switch_provider("local")


@pytest.mark.asyncio
async def test_switch_provider_rejects_unknown_provider() -> None:
    registry = ProviderRegistry()
    engine = AIEngine(
        DummyProvider("openai"),
        provider_registry=registry,
    )

    with pytest.raises(KeyError, match="not registered"):
        await engine.switch_provider("unknown")


@pytest.mark.asyncio
async def test_switch_provider_waits_for_active_generation(
    generation_request: GenerationRequest,
) -> None:
    openai_provider = BlockingProvider("openai")
    local_provider = DummyProvider("local")

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        openai_provider,
        provider_registry=registry,
    )

    generation_task = asyncio.create_task(
        engine.generate(generation_request)
    )

    await asyncio.wait_for(
        openai_provider.started.wait(),
        timeout=1.0,
    )

    switch_task = asyncio.create_task(
        engine.switch_provider("local")
    )

    await asyncio.sleep(0)

    assert not switch_task.done()
    assert engine.provider is openai_provider

    openai_provider.release.set()

    await generation_task
    await switch_task

    assert engine.provider is local_provider


@pytest.mark.asyncio
async def test_concurrent_generations_are_supported(
    generation_request: GenerationRequest,
) -> None:
    provider = ParallelProvider("openai")
    engine = AIEngine(provider)

    first_task = asyncio.create_task(
        engine.generate(generation_request)
    )
    second_task = asyncio.create_task(
        engine.generate(generation_request)
    )

    await asyncio.wait_for(
        provider.both_started.wait(),
        timeout=1.0,
    )

    assert provider.started_count == 2
    assert engine.active_operations == 2

    provider.release.set()

    results = await asyncio.gather(
        first_task,
        second_task,
    )

    assert len(results) == 2
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_switch_does_not_close_previous_provider() -> None:
    openai_provider = DummyProvider("openai")
    local_provider = DummyProvider("local")

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        openai_provider,
        provider_registry=registry,
    )

    await engine.switch_provider("local")

    openai_provider.close_mock.assert_not_awaited()
    local_provider.close_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_closes_all_activated_providers(
    generation_request: GenerationRequest,
) -> None:
    openai_provider = DummyProvider("openai")
    local_provider = DummyProvider("local")

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        openai_provider,
        provider_registry=registry,
    )

    await engine.switch_provider("local")
    await engine.generate(generation_request)
    await engine.switch_provider("openai")

    await engine.close()

    assert engine.is_closed is True

    openai_provider.close_mock.assert_awaited_once()
    local_provider.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_waits_for_active_generation(
    generation_request: GenerationRequest,
) -> None:
    provider = BlockingProvider("openai")
    engine = AIEngine(provider)

    generation_task = asyncio.create_task(
        engine.generate(generation_request)
    )

    await asyncio.wait_for(
        provider.started.wait(),
        timeout=1.0,
    )

    close_task = asyncio.create_task(engine.close())

    await asyncio.sleep(0)

    assert not close_task.done()
    assert engine.is_closed is False

    provider.release.set()

    await generation_task
    await close_task

    assert engine.is_closed is True
    provider.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_waits_for_active_stream(
    generation_request: GenerationRequest,
) -> None:
    provider = BlockingProvider("openai")

    async def blocking_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        provider.started.set()
        await provider.release.wait()
        yield "Completed response."

    provider.stream = blocking_stream

    engine = AIEngine(provider)

    async def consume_stream() -> list[str]:
        return [
            chunk
            async for chunk in engine.stream(generation_request)
        ]

    stream_task = asyncio.create_task(consume_stream())

    await asyncio.wait_for(
        provider.started.wait(),
        timeout=1.0,
    )

    close_task = asyncio.create_task(engine.close())

    await asyncio.sleep(0)

    assert not close_task.done()

    provider.release.set()

    chunks = await stream_task
    await close_task

    assert chunks == ["Completed response."]
    provider.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_is_idempotent_after_provider_switch() -> None:
    openai_provider = DummyProvider("openai")
    local_provider = DummyProvider("local")

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        openai_provider,
        provider_registry=registry,
    )

    await engine.switch_provider("local")
    await engine.close()
    await engine.close()

    openai_provider.close_mock.assert_awaited_once()
    local_provider.close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_switch_rejects_closed_engine() -> None:
    provider = DummyProvider("openai")
    registry = ProviderRegistry()

    engine = AIEngine(
        provider,
        provider_registry=registry,
    )

    await engine.close()

    with pytest.raises(AIProviderUnavailableError):
        await engine.switch_provider("openai")


@pytest.mark.asyncio
async def test_close_attempts_remaining_providers_after_failure() -> None:
    openai_provider = DummyProvider("openai")
    local_provider = DummyProvider("local")

    openai_provider.close_mock.side_effect = RuntimeError(
        "OpenAI cleanup failed."
    )

    registry = ProviderRegistry()
    registry.register(local_provider)

    engine = AIEngine(
        openai_provider,
        provider_registry=registry,
    )

    await engine.switch_provider("local")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await engine.close()

    openai_provider.close_mock.assert_awaited_once()
    local_provider.close_mock.assert_awaited_once()
    assert engine.is_closed is True
