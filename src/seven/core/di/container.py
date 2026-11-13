from pathlib import Path

from dependency_injector import containers, providers

from seven.core.config import AppConfig, ConfigurationManager
from seven.core.events import EventBus
from seven.core.logging import LoggerConfig, LoggerService
from seven.core.runtime import RuntimeManager
from seven.core.settings.settings_manager import SettingsManager
from seven.core.settings.settings_store import SettingsStore


def create_logger_config(
    config_manager: ConfigurationManager,
) -> LoggerConfig:
    config = config_manager.config

    return LoggerConfig(
        level=config.runtime.log_level,
    )


def create_event_logger(logger_service: LoggerService):
    return logger_service.get_logger("EventBus")


class SevenContainer(containers.DeclarativeContainer):
    """Central dependency injection container for Seven."""

    # Configuration Manager
    config_manager = providers.Singleton(
        ConfigurationManager,
        config_path=Path("configs/app.yaml"),
        schema=AppConfig,
        dotenv_path=Path(".env"),
    )

    # Logger Service
    logger_service = providers.Singleton(
        LoggerService,
        config=providers.Callable(
            create_logger_config,
            config_manager,
        ),
    )

    # Event Bus
    event_bus = providers.Singleton(
        EventBus,
        logger=providers.Callable(
            create_event_logger,
            logger_service,
        ),
    )

    # Settings Store
    settings_store = providers.Singleton(
        SettingsStore,
        path=Path("configs/settings.json"),
    )

    # Settings Manager
    settings_manager = providers.Singleton(
        SettingsManager,
        store=settings_store,
        event_bus=event_bus,
    )

    # Runtime Manager
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
    """Stop Seven through Runtime Manager."""

    container.runtime_manager().stop()
