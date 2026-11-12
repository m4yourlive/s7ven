
from seven.core.di import (
    SevenContainer,
    initialize_container,
    shutdown_container,
)

from seven.core.config import ConfigurationManager
from seven.core.logging import LoggerService


def main() -> None:
    container = SevenContainer()

    try:
        print("Initializing Seven DI Container...")

        initialize_container(container)

        config = container.config_manager()
        logger = container.logger_service()

        # Check service instances
        assert isinstance(config, ConfigurationManager)
        assert isinstance(logger, LoggerService)

        # Check singleton behavior
        assert config is container.config_manager()
        assert logger is container.logger_service()

        print("\nSEVEN DI CONTAINER TEST")
        print("-----------------------")
        print("Configuration Manager: OK")
        print("Logger Service: OK")
        print("Singleton Config: OK")
        print("Singleton Logger: OK")
        print("Configuration:", config.config.app.name)
        print("Logger Running:", logger.is_running)

        logger.get_logger("Test").success(
            "Dependency Injection Container works!"
        )

    finally:
        shutdown_container(container)


if __name__ == "__main__":
    main()
