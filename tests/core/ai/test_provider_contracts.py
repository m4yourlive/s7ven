
"""Tests for AI provider contracts and capabilities."""

from collections.abc import AsyncIterator

import pytest

from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)


class DummyProvider(AIProvider):
    """Minimal provider implementation for contract tests."""

    @property
    def name(self) -> str:
        return "dummy"

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
        return None

class LimitedProvider(DummyProvider):
    """Provider with restricted generation capabilities."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_system_messages=True,
            supports_temperature=True,
            supports_max_tokens=True,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        )

def test_provider_capabilities_default_to_supported() -> None:
    capabilities = ProviderCapabilities()

    assert capabilities.supports_streaming is True
    assert capabilities.supports_system_messages is True
    assert capabilities.supports_temperature is True
    assert capabilities.supports_max_tokens is True
    assert capabilities.supports_top_p is True
    assert capabilities.supports_frequency_penalty is True
    assert capabilities.supports_presence_penalty is True

def test_provider_capabilities_accept_restricted_features() -> None:
    provider = LimitedProvider()

    assert provider.capabilities.supports_top_p is False
    assert provider.capabilities.supports_frequency_penalty is False
    assert provider.capabilities.supports_presence_penalty is False
    assert provider.capabilities.supports_streaming is True

@pytest.mark.parametrize(
    "field_name",
    [
        "supports_streaming",
        "supports_system_messages",
        "supports_temperature",
        "supports_max_tokens",
        "supports_top_p",
        "supports_frequency_penalty",
        "supports_presence_penalty",
    ],
)
def test_provider_capabilities_reject_non_boolean_values(
    field_name: str,
) -> None:
    with pytest.raises(TypeError):
        ProviderCapabilities(**{field_name: "true"})

def test_provider_capabilities_are_immutable() -> None:
    capabilities = ProviderCapabilities()

    with pytest.raises(AttributeError):
        capabilities.supports_streaming = False

def test_ai_provider_is_an_abstract_contract() -> None:
    with pytest.raises(TypeError):
        AIProvider()

def test_ai_provider_default_capabilities() -> None:
    provider = DummyProvider()

    assert provider.name == "dummy"
    assert isinstance(provider.capabilities, ProviderCapabilities)
    assert provider.capabilities == ProviderCapabilities()

def test_ai_provider_allows_capability_overrides() -> None:
    provider = LimitedProvider()

    assert provider.capabilities.supports_top_p is False
    assert provider.capabilities.supports_presence_penalty is False