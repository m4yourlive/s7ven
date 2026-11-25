
"""Tests for AI dependency injection."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIConfigurationError
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.di.container import SevenContainer


@pytest.fixture
def container():
    container = SevenContainer()

    fake_config = SimpleNamespace(
        ai=SimpleNamespace(
            provider="openai",
            api_key=SecretStr("test-api-key"),
            model="gpt-4o-mini",
            api_base_url=None,
            request_timeout=60,
        )
    )

    fake_config_manager = SimpleNamespace(config=fake_config)

    container.config_manager.override(fake_config_manager)

    yield container

    container.config_manager.reset_override()


def override_ai_config(
    container: SevenContainer,
    *,
    provider: str,
    api_key: SecretStr | None = SecretStr("test-api-key"),
    api_base_url: str | None = None,
    model: str = "test-model",
) -> None:
    """Override the application AI configuration."""
    container.config_manager.reset_override()

    fake_config = SimpleNamespace(
        ai=SimpleNamespace(
            provider=provider,
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
            request_timeout=60,
        )
    )

    container.config_manager.override(
        SimpleNamespace(config=fake_config)
    )


def test_container_creates_openai_provider(container):
    provider = container.openai_provider()

    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"


def test_container_creates_ai_engine(container):
    engine = container.ai_engine()

    assert isinstance(engine, AIEngine)
    assert engine.provider is container.openai_provider()


def test_container_registers_openai_provider(container):
    registry = container.provider_registry()

    assert registry.has_provider("openai")
    assert "openai" in registry.list_providers()


def test_engine_and_registry_share_provider(container):
    engine = container.ai_engine()
    registry = container.provider_registry()

    assert engine.provider is container.openai_provider()
    assert registry.get("openai") is engine.provider


@pytest.mark.asyncio
async def test_engine_lifecycle_closes_provider(container):
    engine = container.ai_engine()
    provider = engine.provider

    await engine.close()

    assert engine.is_closed is True
    assert provider._closed is True


def test_container_creates_openai_compatible_provider(
    container,
) -> None:
    override_ai_config(
        container,
        provider="openrouter",
        api_base_url="https://openrouter.ai/api/v1",
        model="test-model",
    )

    provider = container.ai_provider()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openrouter"
    assert provider._model == "test-model"


def test_container_selects_compatible_provider_case_insensitively(
    container,
) -> None:
    override_ai_config(
        container,
        provider="OpenRouter",
        api_base_url="https://openrouter.ai/api/v1",
    )

    provider = container.ai_provider()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openrouter"


def test_container_registers_compatible_provider(
    container,
) -> None:
    override_ai_config(
        container,
        provider="groq",
        api_base_url="https://api.groq.com/openai/v1",
        model="test-model",
    )

    registry = container.provider_registry()

    assert registry.has_provider("groq")
    assert registry.get("groq") is container.ai_provider()


def test_engine_uses_selected_compatible_provider(
    container,
) -> None:
    override_ai_config(
        container,
        provider="openrouter",
        api_base_url="https://openrouter.ai/api/v1",
    )

    engine = container.ai_engine()
    registry = container.provider_registry()

    assert isinstance(engine.provider, OpenAICompatibleProvider)
    assert engine.provider.name == "openrouter"
    assert registry.get("openrouter") is engine.provider


def test_container_rejects_missing_compatible_api_key(
    container,
) -> None:
    override_ai_config(
        container,
        provider="openrouter",
        api_key=None,
        api_base_url="https://openrouter.ai/api/v1",
    )

    with pytest.raises(AIConfigurationError, match="API key is missing"):
        container.ai_provider()


def test_container_rejects_missing_compatible_base_url(
    container,
) -> None:
    override_ai_config(
        container,
        provider="openrouter",
        api_base_url=None,
    )

    with pytest.raises(
        AIConfigurationError,
        match="API base URL is missing",
    ):
        container.ai_provider()


def test_container_rejects_invalid_compatible_base_url(
    container,
) -> None:
    override_ai_config(
        container,
        provider="openrouter",
        api_base_url="invalid-url",
    )

    with pytest.raises(
        AIConfigurationError,
        match="Invalid configuration",
    ):
        container.ai_provider()
