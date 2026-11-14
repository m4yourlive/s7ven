from seven.core.config import ConfigurationManager
from seven.core.di import (
    SevenContainer,
    initialize_container,
    shutdown_container,
)
from seven.core.events import EventBus
from seven.core.logging import LoggerService
from seven.core.runtime import RuntimeManager, RuntimeState


def main() -> None:
    container = SevenContainer()

    try:
        print("Initializing Seven...")

        initialize_container(container)

        config = container.config_manager()
        logger = container.logger_service()
        event_bus = container.event_bus()
        runtime = container.runtime_manager()

        assert isinstance(config, ConfigurationManager)
        assert isinstance(logger, LoggerService)
        assert isinstance(event_bus, EventBus)
        assert isinstance(runtime, RuntimeManager)

        # Singleton checks
        assert config is container.config_manager()
        assert logger is container.logger_service()
        assert event_bus is container.event_bus()
        assert runtime is container.runtime_manager()

        # Runtime check
        assert runtime.state == RuntimeState.RUNNING
        assert logger.is_running

        print("\nSEVEN RUNTIME TEST")
        print("------------------")
        print("Configuration Manager: OK")
        print("Logger Service: OK")
        print("Event Bus: OK")
        print("Runtime Manager: OK")
        print("Singleton Services: OK")
        print("Runtime State:", runtime.state.value)

        logger.get_logger("Test").success(
            "Seven runtime initialized successfully!"
        )

    finally:
        shutdown_container(container)

    assert runtime.state == RuntimeState.STOPPED
    print("Runtime Shutdown: OK")

if __name__ == "__main__":
    main()