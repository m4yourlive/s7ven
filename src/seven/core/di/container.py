"""Dependency injection container for S7VEN."""

from pathlib import Path

from dependency_injector import containers, providers

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIConfigurationError
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.config import AppConfig, ConfigurationManager
from seven.core.config.config_schema import AIConfig
from seven.core.events import EventBus
from seven.core.logging import LoggerConfig, LoggerService
from seven.core.runtime import RuntimeManager
from seven.core.settings.settings_manager import SettingsManager
from seven.core.settings.settings_store import SettingsStore
from seven.core.ai.prompts import (
    DEFAULT_PROMPTS,
    PromptManager,
)

from seven.core.ai.pipeline import AIResponsePipeline

def create_logger_config(
    config_manager: ConfigurationManager,
) -> LoggerConfig:
    config = config_manager.config

    return LoggerConfig(level=config.runtime.log_level)


def create_event_logger(logger_service: LoggerService):
    return logger_service.get_logger("EventBus")


def create_ai_config(
    config_manager: ConfigurationManager,
) -> AIConfig:
    """Retrieve AI configuration from the configuration manager."""
    return config_manager.config.ai


def create_openai_provider(
    config_manager: ConfigurationManager,
) -> OpenAIProvider:
    """Create OpenAI provider from application configuration."""

    config = config_manager.config.ai

    if config.provider.lower() != "openai":
        raise AIConfigurationError(
            f"Unsupported AI provider: {config.provider}"
        )

    if config.api_key is None:
        raise AIConfigurationError(
            "OpenAI API key is missing. "
            "Configure SEVEN_AI__API_KEY in .env."
        )

    return OpenAIProvider(
        api_key=config.api_key.get_secret_value(),
        model=config.model,
        base_url=config.api_base_url,
        timeout=config.request_timeout,
    )


def create_provider_registry(
    openai_provider: OpenAIProvider,
) -> ProviderRegistry:
    """Create registry and register available AI providers."""

    registry = ProviderRegistry()
    registry.register(openai_provider)

    return registry


class SevenContainer(containers.DeclarativeContainer):
    """Central dependency injection container for Seven."""

    config_manager = providers.Singleton(
        ConfigurationManager,
        config_path=Path("configs/app.yaml"),
        schema=AppConfig,
        dotenv_path=Path(".env"),
    )

    logger_service = providers.Singleton(
        LoggerService,
        config=providers.Callable(
            create_logger_config,
            config_manager,
        ),
    )

    event_bus = providers.Singleton(
        EventBus,
        logger=providers.Callable(
            create_event_logger,
            logger_service,
        ),
    )

    settings_store = providers.Singleton(
        SettingsStore,
        path=Path("configs/settings.json"),
    )

    settings_manager = providers.Singleton(
        SettingsManager,
        store=settings_store,
        event_bus=event_bus,
    )

    ai_config = providers.Singleton(
        create_ai_config,
        config_manager,
    )

    openai_provider = providers.Singleton(
        create_openai_provider,
        config_manager,
    )

    provider_registry = providers.Singleton(
        create_provider_registry,
        openai_provider=openai_provider,
    )

        # Prompt Manager
    prompt_manager = providers.Singleton(
        PromptManager,
        templates=DEFAULT_PROMPTS,
    )

    generation_request_factory = providers.Singleton(
        GenerationRequestFactory,
        config=ai_config,
        prompt_manager=prompt_manager,
    )

    ai_engine = providers.Singleton(
        AIEngine,
        provider=openai_provider,
    )

        # AI Response Pipeline
    ai_pipeline = providers.Singleton(
        AIResponsePipeline,
        engine=ai_engine,
        request_factory=generation_request_factory,
    )

    runtime_manager = providers.Singleton(
        RuntimeManager,
        config_manager=config_manager,
        logger_service_provider=providers.Delegate(
            logger_service
        ),
        event_bus_provider=providers.Delegate(
            event_bus
        ),
    )


def initialize_container(container: SevenContainer) -> None:
    """Start Seven through Runtime Manager."""
    container.runtime_manager().start()


def shutdown_container(container: SevenContainer) -> None:
    """Stop Seven and release initialized AI resources."""
    import asyncio

    try:
        ai_engine_provider = getattr(container, "ai_engine", None)

        if ai_engine_provider is not None:
            engine = ai_engine_provider()
            asyncio.run(engine.close())
    finally:
        container.runtime_manager().stop()
