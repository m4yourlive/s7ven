
"""Dependency injection container for S7VEN."""

import asyncio
from pathlib import Path

from dependency_injector import containers, providers

from seven.core.ai.context_manager import ConversationContextManager
from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIConfigurationError
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory

from seven.core.config import AppConfig, ConfigurationManager
from seven.core.config.config_schema import AIConfig

from seven.core.conversations import (
    ConversationService,
    MemoryConsolidator,
    MemoryService,
    MessageService,
)

from seven.core.database import DatabaseConfig, DatabaseSessionManager
from seven.core.database.repositories import (
    ConversationRepository,
    MessageRepository,
    MemoryRepository,
)

from seven.core.events import EventBus
from seven.core.logging import LoggerConfig, LoggerService
from seven.core.runtime import RuntimeManager
from seven.core.settings.settings_manager import SettingsManager
from seven.core.settings.settings_store import SettingsStore

from seven.core.voice.assistant import VoiceAssistant
from seven.core.voice.exceptions import VoiceProviderUnavailableError
from seven.core.voice.pipeline import VoicePipeline
from seven.core.voice.providers.openai_provider import OpenAIVoiceProvider


def create_logger_config(
    config_manager: ConfigurationManager,
) -> LoggerConfig:
    """Create logger configuration."""
    config = config_manager.config

    return LoggerConfig(level=config.runtime.log_level)


def create_event_logger(logger_service: LoggerService):
    """Create the EventBus logger."""
    return logger_service.get_logger("EventBus")


def create_ai_config(
    config_manager: ConfigurationManager,
) -> AIConfig:
    """Retrieve AI configuration."""
    return config_manager.config.ai


def create_database_config(
    config_manager: ConfigurationManager,
) -> DatabaseConfig:
    """Create database configuration from application settings."""
    return DatabaseConfig.from_storage_config(
        config_manager.config.storage,
    )


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


def create_openai_voice_provider(
    config_manager: ConfigurationManager,
) -> OpenAIVoiceProvider:
    """Create OpenAI voice provider from application configuration."""
    config = config_manager.config.ai

    if config.api_key is None:
        raise VoiceProviderUnavailableError(
            "OpenAI API key is required for voice interaction."
        )

    return OpenAIVoiceProvider(
        api_key=config.api_key.get_secret_value(),
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

    # Configuration
    config_manager = providers.Singleton(
        ConfigurationManager,
        config_path=Path("configs/app.yaml"),
        schema=AppConfig,
        dotenv_path=Path(".env"),
    )

    # Logging and Events
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

    # Settings
    settings_store = providers.Singleton(
        SettingsStore,
        path=Path("configs/settings.json"),
    )

    settings_manager = providers.Singleton(
        SettingsManager,
        store=settings_store,
        event_bus=event_bus,
    )

    # Database
    database_config = providers.Singleton(
        create_database_config,
        config_manager,
    )

    database_session_manager = providers.Singleton(
        DatabaseSessionManager,
        config=database_config,
    )

    # Repositories
    conversation_repository = providers.Factory(
        ConversationRepository,
    )

    message_repository = providers.Factory(
        MessageRepository,
    )

    memory_repository = providers.Factory(
        MemoryRepository,
    )

    # Conversation and Memory Services
    conversation_service = providers.Factory(
        ConversationService,
    )

    message_service = providers.Factory(
        MessageService,
    )

    memory_service = providers.Factory(
        MemoryService,
    )

    memory_consolidator = providers.Factory(
        MemoryConsolidator,
        message_service=message_service,
        memory_service=memory_service,
    )

    # AI Configuration
    ai_config = providers.Singleton(
        create_ai_config,
        config_manager,
    )

    # AI Providers
    openai_provider = providers.Singleton(
        create_openai_provider,
        config_manager,
    )

    provider_registry = providers.Singleton(
        create_provider_registry,
        openai_provider=openai_provider,
    )

    # Prompt Management
    prompt_manager = providers.Singleton(
        PromptManager,
        templates=DEFAULT_PROMPTS,
    )

    generation_request_factory = providers.Singleton(
        GenerationRequestFactory,
        config=ai_config,
        prompt_manager=prompt_manager,
    )

    # AI Engine
    ai_engine = providers.Singleton(
        AIEngine,
        provider=openai_provider,
    )

    conversation_context_manager = providers.Singleton(
        ConversationContextManager,
        max_messages=20,
    )

    ai_pipeline = providers.Singleton(
        AIResponsePipeline,
        engine=ai_engine,
        request_factory=generation_request_factory,
        context_manager=conversation_context_manager,
    )

    # Voice Providers
    openai_voice_provider = providers.Singleton(
        create_openai_voice_provider,
        config_manager,
    )

    voice_pipeline = providers.Singleton(
        VoicePipeline,
        stt_provider=openai_voice_provider,
        tts_provider=openai_voice_provider,
    )

    voice_assistant = providers.Singleton(
        VoiceAssistant,
        voice_pipeline=voice_pipeline,
        ai_pipeline=ai_pipeline,
    )

    # Runtime
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
    """Stop Seven and release initialized resources."""

    async def close_resources() -> None:
        voice_provider = getattr(
            container,
            "voice_pipeline",
            None,
        )

        if (
            voice_provider is not None
            and voice_provider.is_initialized()
        ):
            await voice_provider().close()

        ai_engine_provider = getattr(
            container,
            "ai_engine",
            None,
        )

        if (
            ai_engine_provider is not None
            and ai_engine_provider.is_initialized()
        ):
            await ai_engine_provider().close()

        database_provider = getattr(
            container,
            "database_session_manager",
            None,
        )

        if (
            database_provider is not None
            and database_provider.is_initialized()
        ):
            await database_provider().dispose()

    try:
        asyncio.run(close_resources())
    finally:
        container.runtime_manager().stop()
