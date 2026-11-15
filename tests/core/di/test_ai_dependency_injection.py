from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from seven.core.ai.engine import AIEngine
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
    assert registry.has_provider(engine.provider.name)


@pytest.mark.asyncio
async def test_engine_lifecycle_closes_provider(container):
    engine = container.ai_engine()
    provider = engine.provider

    await engine.close()

    assert engine.is_closed is True
    assert provider._closed is True
