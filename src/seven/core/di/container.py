
"""Dependency injection container for S7VEN."""

import asyncio
from pathlib import Path

from dependency_injector import containers, providers

from seven.core.ai.context_manager import ConversationContextManager
from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIConfigurationError
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.ai.runtimes import LocalRuntimeFactory, OllamaRuntime

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
from seven.core.integrations import IntegrationServiceManager
from seven.core.logging import LoggerConfig, LoggerService
from seven.core.runtime import RuntimeManager
from seven.core.settings.settings_manager import SettingsManager
from seven.core.settings.settings_store import SettingsStore

from seven.core.voice.assistant import VoiceAssistant
from seven.core.voice.exceptions import VoiceProviderUnavailableError
from seven.core.voice.pipeline import VoicePipeline
from seven.core.voice.providers.openai_provider import OpenAIVoiceProvider

from seven.core.ai.configuration_coordinator import (
    AIConfigurationCoordinator,
)

from seven.core.integrations.system import SystemInformationService

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


def get_ai_provider_mode(
    config_manager: ConfigurationManager,
) -> str:
    """Resolve whether the application uses API or local AI."""
    configured_mode = getattr(config_manager.config.ai, "mode", "api")
    mode = configured_mode.strip().casefold()

    if mode not in {"api", "local"}:
        raise AIConfigurationError(
            f"Unsupported AI mode {mode!r}. Expected 'api' or 'local'."
        )

    return mode


def get_ai_provider_kind(
    config_manager: ConfigurationManager,
) -> str:
    """Resolve the provider implementation from AI configuration."""
    provider_name = config_manager.config.ai.provider.strip().casefold()

    if provider_name == "openai":
        return "openai"

    return "openai_compatible"


def create_openai_provider(
    config_manager: ConfigurationManager,
) -> OpenAIProvider:
    """Create the native OpenAI provider from application configuration."""
    config = config_manager.config.ai

    if config.provider.strip().casefold() != "openai":
        raise AIConfigurationError(
            "OpenAIProvider requires AI provider configuration "
            "to be set to 'openai'."
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


def create_openai_compatible_provider(
    config_manager: ConfigurationManager,
) -> OpenAICompatibleProvider:
    """Create a third-party OpenAI-compatible provider."""
    config = config_manager.config.ai
    provider_name = config.provider.strip().casefold()

    if provider_name == "openai":
        raise AIConfigurationError(
            "Use the native OpenAI provider for OpenAI configuration."
        )

    if config.api_key is None:
        raise AIConfigurationError(
            f"API key is missing for provider '{provider_name}'. "
            "Configure SEVEN_AI__API_KEY in .env."
        )

    if not config.api_base_url:
        raise AIConfigurationError(
            f"API base URL is missing for provider '{provider_name}'. "
            "Configure SEVEN_AI__API_BASE_URL in .env."
        )

    try:
        return OpenAICompatibleProvider(
            api_key=config.api_key.get_secret_value(),
            provider_name=provider_name,
            base_url=config.api_base_url,
            model=config.model,
            timeout=config.request_timeout,
        )
    except (TypeError, ValueError) as exc:
        raise AIConfigurationError(
            f"Invalid configuration for provider '{provider_name}': {exc}"
        ) from exc


def create_local_runtime_factory() -> LocalRuntimeFactory:
    """Create the registry of supported local runtime builders."""
    factory = LocalRuntimeFactory()
    factory.register("ollama", OllamaRuntime)
    return factory


def create_local_ai_provider(
    config_manager: ConfigurationManager,
    runtime_factory: LocalRuntimeFactory,
) -> AIProvider:
    """Create a local AI provider from the selected runtime configuration."""
    config = config_manager.config.ai

    if config.mode.strip().casefold() != "local":
        raise AIConfigurationError(
            "LocalAIProvider requires AI mode to be set to 'local'."
        )

    try:
        return runtime_factory.create_provider(
            config=config,
            provider_name="local",
        )
    except (TypeError, ValueError) as exc:
        raise AIConfigurationError(
            f"Invalid local AI configuration: {exc}"
        ) from exc


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
    ai_provider: AIProvider,
) -> ProviderRegistry:
    """Create a registry and register the configured AI provider."""
    registry = ProviderRegistry()
    registry.register(ai_provider)

    return registry


def get_active_ai_provider(engine: AIEngine) -> AIProvider:
    """Resolve the provider currently managed by the AI engine."""
    return engine.provider


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
    ai_config = providers.Callable(
        create_ai_config,
        config_manager,
    )

    # AI Providers
    openai_provider = providers.Singleton(
        create_openai_provider,
        config_manager,
    )

    openai_compatible_provider = providers.Singleton(
        create_openai_compatible_provider,
        config_manager,
    )

    api_ai_provider = providers.Selector(
        providers.Callable(
            get_ai_provider_kind,
            config_manager,
        ),
        openai=openai_provider,
        openai_compatible=openai_compatible_provider,
    )

    local_runtime_factory = providers.Singleton(
        create_local_runtime_factory,
    )

    local_ai_provider = providers.Singleton(
        create_local_ai_provider,
        config_manager=config_manager,
        runtime_factory=local_runtime_factory,
    )

    initial_ai_provider = providers.Selector(
        providers.Callable(
            get_ai_provider_mode,
            config_manager,
        ),
        api=api_ai_provider,
        local=local_ai_provider,
    )

    provider_registry = providers.Singleton(
        create_provider_registry,
        ai_provider=initial_ai_provider,
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
        provider=initial_ai_provider,
        provider_registry=provider_registry,
    )

    # Public provider access always reflects the engine's active provider.
    ai_provider = providers.Callable(
        get_active_ai_provider,
        engine=ai_engine,
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

    # AI Configuration Coordination
    ai_configuration_coordinator = providers.Singleton(
        AIConfigurationCoordinator,
        config_manager=config_manager,
        engine=ai_engine,
        request_factory=generation_request_factory,
        local_runtime_factory=local_runtime_factory,
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

    # System Integrations
    system_information_service = providers.Singleton(
        SystemInformationService,
    )

    integration_service_manager = providers.Singleton(
        IntegrationServiceManager,
        services=providers.List(
            system_information_service,
        ),
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
        integration_manager_provider=providers.Delegate(
            integration_service_manager
        ),
    )

def initialize_container(container: SevenContainer) -> None:
    """Start Seven through Runtime Manager."""
    container.runtime_manager().start()




def shutdown_container(container: SevenContainer) -> None:
    """Stop Seven and release container resources."""

    async def close_resources() -> None:
        resources = (
            ("voice_pipeline", "close"),
            ("ai_engine", "close"),
            ("database_session_manager", "dispose"),
        )

        first_error: Exception | None = None

        for provider_name, method_name in resources:
            resource_provider = getattr(
                container,
                provider_name,
                None,
            )

            if resource_provider is None:
                continue

            try:
                resource = resource_provider()
                cleanup = getattr(resource, method_name, None)

                if callable(cleanup):
                    await cleanup()

            except Exception as error:
                if first_error is None:
                    first_error = error

        if first_error is not None:
            raise first_error

    try:
        asyncio.run(close_resources())
    finally:
        container.runtime_manager().stop()
