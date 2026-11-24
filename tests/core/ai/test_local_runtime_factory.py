
"""Tests for local runtime factory behavior."""

from unittest.mock import AsyncMock

import pytest

from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import GenerationRequest, GenerationResponse
from seven.core.ai.providers.local_ai_provider import LocalAIProvider
from seven.core.ai.runtimes import LocalRuntimeFactory
from seven.core.config.config_schema import AIConfig


class FakeLocalRuntime(LocalModelRuntime):
    """Test implementation of a local model runtime."""

    def __init__(self, runtime_name: str = "ollama") -> None:
        self._runtime_name = runtime_name
        self.generate_mock = AsyncMock()
        self.stream_mock = AsyncMock()
        self.close_mock = AsyncMock()

    @property
    def runtime_name(self) -> str:
        return self._runtime_name

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return await self.generate_mock(request)

    async def stream(self, request: GenerationRequest):
        chunks = await self.stream_mock(request)

        for chunk in chunks:
            yield chunk

    async def close(self) -> None:
        await self.close_mock()


@pytest.fixture
def config() -> AIConfig:
    return AIConfig(
        mode="local",
        local_runtime="ollama",
        local_model="llama3.2",
    )


@pytest.fixture
def runtime() -> FakeLocalRuntime:
    return FakeLocalRuntime()


@pytest.fixture
def factory(runtime: FakeLocalRuntime) -> LocalRuntimeFactory:
    instance = LocalRuntimeFactory()
    instance.register("ollama", lambda config: runtime)
    return instance


def test_register_runtime_builder(factory: LocalRuntimeFactory) -> None:
    assert factory.has_runtime("ollama") is True
    assert "ollama" in factory
    assert len(factory) == 1


def test_register_normalizes_runtime_name() -> None:
    factory = LocalRuntimeFactory()
    factory.register("  OLLAMA  ", lambda config: FakeLocalRuntime())

    assert factory.has_runtime("ollama") is True
    assert factory.list_runtimes() == ["ollama"]


def test_register_rejects_duplicate_builder(
    factory: LocalRuntimeFactory,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        factory.register("ollama", lambda config: FakeLocalRuntime())


def test_register_rejects_invalid_builder() -> None:
    factory = LocalRuntimeFactory()

    with pytest.raises(TypeError, match="callable"):
        factory.register("ollama", None)


def test_unregister_runtime_builder(
    factory: LocalRuntimeFactory,
) -> None:
    builder = factory.unregister("ollama")

    assert callable(builder)
    assert factory.has_runtime("ollama") is False
    assert len(factory) == 0


def test_unregister_rejects_unknown_runtime() -> None:
    factory = LocalRuntimeFactory()

    with pytest.raises(KeyError, match="not registered"):
        factory.unregister("ollama")


def test_create_runtime(
    factory: LocalRuntimeFactory,
    config: AIConfig,
    runtime: FakeLocalRuntime,
) -> None:
    result = factory.create_runtime(config)

    assert result is runtime
    assert result.runtime_name == "ollama"


def test_create_runtime_rejects_unknown_runtime(
    config: AIConfig,
) -> None:
    factory = LocalRuntimeFactory()

    with pytest.raises(ValueError, match="not registered"):
        factory.create_runtime(config)


def test_create_runtime_rejects_invalid_config() -> None:
    factory = LocalRuntimeFactory()

    with pytest.raises(TypeError, match="AIConfig"):
        factory.create_runtime(None)


def test_create_runtime_rejects_invalid_builder_result(
    config: AIConfig,
) -> None:
    factory = LocalRuntimeFactory()
    factory.register("ollama", lambda config: object())

    with pytest.raises(TypeError, match="LocalModelRuntime"):
        factory.create_runtime(config)


def test_create_runtime_rejects_mismatched_runtime_name(
    config: AIConfig,
) -> None:
    factory = LocalRuntimeFactory()
    factory.register(
        "ollama",
        lambda config: FakeLocalRuntime("llama.cpp"),
    )

    with pytest.raises(ValueError, match="returned runtime"):
        factory.create_runtime(config)


def test_create_provider(
    factory: LocalRuntimeFactory,
    config: AIConfig,
    runtime: FakeLocalRuntime,
) -> None:
    provider = factory.create_provider(config)

    assert isinstance(provider, LocalAIProvider)
    assert provider.name == "local"
    assert provider.runtime is runtime


def test_create_provider_supports_custom_provider_name(
    factory: LocalRuntimeFactory,
    config: AIConfig,
) -> None:
    provider = factory.create_provider(
        config,
        provider_name="local-ollama",
    )

    assert provider.name == "local-ollama"


def test_create_provider_uses_configured_runtime(
    runtime: FakeLocalRuntime,
) -> None:
    factory = LocalRuntimeFactory()
    factory.register("ollama", lambda config: runtime)

    config = AIConfig(
        mode="local",
        local_runtime="ollama",
    )

    provider = factory.create_provider(config)

    assert provider.runtime is runtime
