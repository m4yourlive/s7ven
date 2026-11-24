
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from seven.core.ai.configuration_coordinator import (
    AIConfigurationCoordinator,
)
from seven.core.ai.engine import AIEngine
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.config import AppConfig, ConfigurationManager
from seven.core.di.container import SevenContainer
from seven.core.ai.exceptions import AIConfigurationError

def write_config(path: Path, provider: str) -> None:
    config = {
        "ai": {
            "mode": "api",
            "provider": provider,
            "api_key": "test-api-key",
            "model": "test-model",
            "api_base_url": (
                "https://openrouter.ai/api/v1"
                if provider == "openrouter"
                else None
            ),
            "request_timeout": 60,
        },
        "runtime": {"log_level": "INFO"},
        "storage": {},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


@pytest.fixture
def reload_container(tmp_path: Path):
    config_path = tmp_path / "app.yaml"
    write_config(config_path, "openai")

    container = SevenContainer()

    config_manager = ConfigurationManager(
        config_path=config_path,
        schema=AppConfig,
        dotenv_path=tmp_path / ".env",
    )

    container.config_manager.override(config_manager)
    config_manager.load()

    yield container, config_path

    engine = container.ai_engine()

    if not engine.is_closed:
        asyncio.run(engine.close())

    container.config_manager.reset_override()


def test_ai_dependency_graph_uses_shared_instances(
    reload_container,
) -> None:
    container, _ = reload_container

    engine = container.ai_engine()
    registry = container.provider_registry()
    provider = container.ai_provider()

    assert isinstance(engine, AIEngine)
    assert engine.provider is provider
    assert registry.get(provider.name) is provider


def test_configuration_coordinator_is_singleton(
    reload_container,
) -> None:
    container, _ = reload_container

    first = container.ai_configuration_coordinator()
    second = container.ai_configuration_coordinator()

    assert isinstance(first, AIConfigurationCoordinator)
    assert first is second


def test_coordinator_uses_container_ai_dependencies(
    reload_container,
) -> None:
    container, _ = reload_container

    coordinator = container.ai_configuration_coordinator()

    assert coordinator.engine is container.ai_engine()
    assert coordinator.request_factory is (
        container.generation_request_factory()
    )
    assert coordinator.config_manager is container.config_manager()


def test_ai_config_provider_reflects_active_configuration(
    reload_container,
) -> None:
    container, config_path = reload_container

    initial_config = container.ai_config()

    assert initial_config.provider == "openai"

    write_config(config_path, "openrouter")

    coordinator = container.ai_configuration_coordinator()
    asyncio.run(coordinator.reload())

    updated_config = container.ai_config()

    assert updated_config.provider == "openrouter"
    assert updated_config.model == "test-model"


def test_ai_provider_matches_engine_after_reload(
    reload_container,
) -> None:
    container, config_path = reload_container

    engine = container.ai_engine()
    coordinator = container.ai_configuration_coordinator()

    assert engine.provider.name == "openai"

    write_config(config_path, "openrouter")
    asyncio.run(coordinator.reload())

    assert isinstance(
        engine.provider,
        OpenAICompatibleProvider,
    )
    assert engine.provider.name == "openrouter"
    assert engine.provider is coordinator.active_provider


def test_container_provider_matches_engine_after_reload(
    reload_container,
) -> None:
    container, config_path = reload_container

    engine = container.ai_engine()
    coordinator = container.ai_configuration_coordinator()

    write_config(config_path, "openrouter")
    asyncio.run(coordinator.reload())

    active_provider = engine.provider

    assert container.ai_provider() is active_provider
    assert container.provider_registry().get(
        active_provider.name
    ) is active_provider


def test_ai_pipeline_uses_shared_dependencies(
    reload_container,
) -> None:
    container, _ = reload_container

    pipeline = container.ai_pipeline()

    assert pipeline._engine is container.ai_engine()
    assert pipeline._request_factory is (
        container.generation_request_factory()
    )


def test_pipeline_observes_provider_switch(
    reload_container,
) -> None:
    container, config_path = reload_container

    pipeline = container.ai_pipeline()
    engine = container.ai_engine()
    original_engine = pipeline._engine

    assert original_engine is engine
    assert engine.provider.name == "openai"

    write_config(config_path, "openrouter")
    asyncio.run(
        container.ai_configuration_coordinator().reload()
    )

    assert pipeline._engine is engine
    assert pipeline._engine.provider is engine.provider
    assert pipeline._engine.provider.name == "openrouter"


def test_coordinator_and_pipeline_share_request_factory(
    reload_container,
) -> None:
    container, _ = reload_container

    coordinator = container.ai_configuration_coordinator()
    pipeline = container.ai_pipeline()

    assert coordinator.request_factory is (
        pipeline._request_factory
    )
    assert coordinator.request_factory is (
        container.generation_request_factory()
    )


@pytest.mark.asyncio
async def test_engine_closes_all_providers_after_reload(
    reload_container,
) -> None:
    container, config_path = reload_container

    engine = container.ai_engine()
    coordinator = container.ai_configuration_coordinator()

    initial_provider = engine.provider

    write_config(config_path, "openrouter")
    await coordinator.reload()

    active_provider = engine.provider

    assert active_provider is not initial_provider

    await engine.close()

    assert engine.is_closed is True
    assert initial_provider._closed is True
    assert active_provider._closed is True


@pytest.mark.asyncio
async def test_engine_close_is_idempotent(
    reload_container,
) -> None:
    container, _ = reload_container

    engine = container.ai_engine()
    provider = engine.provider

    await engine.close()
    await engine.close()

    assert engine.is_closed is True
    assert provider._closed is True


@pytest.mark.asyncio
async def test_coordinator_rejects_reload_after_engine_shutdown(
    reload_container,
) -> None:
    container, config_path = reload_container

    engine = container.ai_engine()
    coordinator = container.ai_configuration_coordinator()

    await engine.close()

    write_config(config_path, "openrouter")

    with pytest.raises(
        AIConfigurationError,
        match="closed",
    ):
        await coordinator.reload()

    assert engine.is_closed is True
