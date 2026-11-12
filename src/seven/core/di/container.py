from pathlib import Path

from dependency_injector import containers, providers

from seven.core.config import AppConfig, ConfigurationManager
from seven.core.logging import LoggerConfig, LoggerService


def create_logger_config(
    config_manager: ConfigurationManager,
) -> LoggerConfig:
    config = config_manager.config

    return LoggerConfig(
        level=config.runtime.log_level,
    )


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


def initialize_container(container: SevenContainer) -> None:
    """Initialize Seven core services."""

    config_manager = container.config_manager()
    config_manager.load()

    logger_service = container.logger_service()
    logger_service.start()

    logger_service.get_logger("DIContainer").info(
        "Seven dependency container initialized"
    )



def shutdown_container(container: SevenContainer) -> None:
    """Gracefully shut down Seven core services."""

    # Jangan membuat Logger jika konfigurasi gagal dimuat.
    config_manager = container.config_manager()

    if not config_manager.is_loaded:
        return

    logger_service = container.logger_service()

    if logger_service.is_running:
        logger_service.get_logger("DIContainer").info(
            "Shutting down Seven dependency container"
        )
        logger_service.stop()
