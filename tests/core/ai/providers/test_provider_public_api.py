"""Tests for the public AI provider package API."""

import seven.core.ai.providers as providers
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.providers.registry import ProviderRegistry


def test_provider_package_exports_base_types() -> None:
    assert providers.AIProvider is AIProvider
    assert providers.ProviderRegistry is ProviderRegistry


def test_provider_package_exports_api_providers() -> None:
    assert providers.OpenAIProvider is OpenAIProvider
    assert providers.OpenAICompatibleProvider is OpenAICompatibleProvider


def test_openai_compatible_provider_inherits_openai_provider() -> None:
    assert issubclass(OpenAICompatibleProvider, OpenAIProvider)


def test_provider_package_declares_public_exports() -> None:
    assert providers.__all__ == [
        "AIProvider",
        "OpenAICompatibleProvider",
        "OpenAIProvider",
        "ProviderRegistry",
    ]


def test_provider_package_directory_contains_public_exports() -> None:
    module_attributes = dir(providers)

    for name in providers.__all__:
        assert name in module_attributes


def test_unknown_provider_attribute_raises_attribute_error() -> None:
    try:
        getattr(providers, "UnknownProvider")
    except AttributeError:
        pass
    else:
        raise AssertionError("Unknown provider attribute should raise AttributeError")