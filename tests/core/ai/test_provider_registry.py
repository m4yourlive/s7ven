
"""Tests for the AI provider registry."""

from collections.abc import AsyncIterator

import pytest

from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)
from seven.core.ai.providers import ProviderRegistry


class DummyProvider(AIProvider):
    """Minimal provider implementation for registry tests."""

    def __init__(
        self,
        provider_name: str = "dummy",
    ) -> None:
        self._name = provider_name
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return GenerationResponse(
            content="Test response.",
            model=request.model,
            provider=self.name,
        )

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Test response."

    async def close(self) -> None:
        self.closed = True

class InvalidCapabilitiesProvider(DummyProvider):
    """Provider returning invalid capability metadata."""

    @property
    def capabilities(self):
        return "invalid"

@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()

@pytest.fixture
def provider() -> DummyProvider:
    return DummyProvider()

def test_register_provider(
    registry: ProviderRegistry,
    provider: DummyProvider,
) -> None:
    registry.register(provider)

    assert registry.has_provider("dummy")
    assert registry.get("dummy") is provider

def test_register_normalizes_provider_name(
    registry: ProviderRegistry,
) -> None:
    provider = DummyProvider("  Local-AI  ")

    registry.register(provider)

    assert registry.has_provider("local-ai")
    assert registry.has_provider("LOCAL-AI")
    assert registry.get("LOCAL-AI") is provider

def test_register_rejects_non_provider(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]

def test_register_rejects_empty_provider_name(
    registry: ProviderRegistry,
) -> None:
    provider = DummyProvider(" ")

    with pytest.raises(ValueError):
        registry.register(provider)

def test_register_rejects_duplicate_provider(
    registry: ProviderRegistry,
) -> None:
    registry.register(DummyProvider("openai"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyProvider("OPENAI"))

def test_register_rejects_invalid_capabilities(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(TypeError, match="ProviderCapabilities"):
        registry.register(InvalidCapabilitiesProvider())

def test_get_rejects_unknown_provider(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get("unknown")

def test_get_rejects_invalid_name(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.get(" ")

def test_get_rejects_non_string_name(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.get(None)  # type: ignore[arg-type]

def test_unregister_provider(
    registry: ProviderRegistry,
    provider: DummyProvider,
) -> None:
    registry.register(provider)

    removed = registry.unregister("dummy")

    assert removed is provider
    assert not registry.has_provider("dummy")

def test_unregister_rejects_unknown_provider(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister("unknown")

def test_list_providers_preserves_registration_order(
    registry: ProviderRegistry,
) -> None:
    registry.register(DummyProvider("openai"))
    registry.register(DummyProvider("local"))

    assert registry.list_providers() == ("openai", "local")

def test_get_capabilities(
    registry: ProviderRegistry,
) -> None:
    provider = DummyProvider()

    registry.register(provider)

    capabilities = registry.get_capabilities("dummy")

    assert isinstance(capabilities, ProviderCapabilities)
    assert capabilities == provider.capabilities

def test_provider_membership(
    registry: ProviderRegistry,
) -> None:
    registry.register(DummyProvider("openai"))

    assert "openai" in registry
    assert "OPENAI" in registry
    assert "unknown" not in registry
    assert object() not in registry

def test_registry_length(
    registry: ProviderRegistry,
) -> None:
    assert len(registry) == 0

    registry.register(DummyProvider("openai"))
    registry.register(DummyProvider("local"))

    assert len(registry) == 2

    registry.unregister("openai")

    assert len(registry) == 1