from collections.abc import Callable
from threading import RLock

from seven.core.config import ConfigurationManager
from seven.core.events import EventBus
from seven.core.logging import LoggerService
from seven.core.runtime.runtime_events import (
    RuntimeFailed,
    RuntimeStarted,
    RuntimeStarting,
    RuntimeStopped,
    RuntimeStopping,
)
from seven.core.runtime.runtime_state import RuntimeState


class RuntimeManager:
    """
    Controls the lifecycle of the Seven application.

    Responsibilities:
    - Application startup
    - Core service initialization
    - Runtime state management
    - Lifecycle event publishing
    - Graceful shutdown
    - Startup failure handling
    """

    def __init__(
        self,
        config_manager: ConfigurationManager,
        logger_service_provider: Callable[[], LoggerService],
        event_bus_provider: Callable[[], EventBus],
    ):
        self._config_manager = config_manager
        self._logger_service_provider = logger_service_provider
        self._event_bus_provider = event_bus_provider

        self._logger_service: LoggerService | None = None
        self._event_bus: EventBus | None = None

        self._state = RuntimeState.STOPPED
        self._lock = RLock()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == RuntimeState.RUNNING

    def start(self) -> bool:
        """Initialize core services and start Seven."""

        with self._lock:
            if self._state == RuntimeState.RUNNING:
                return False

            if self._state in (
                RuntimeState.STARTING,
                RuntimeState.STOPPING,
            ):
                raise RuntimeError(
                    f"Cannot start while runtime is {self._state.value}"
                )

            self._state = RuntimeState.STARTING

            try:
                # 1. Load configuration
                if not self._config_manager.is_loaded:
                    self._config_manager.load()

                # 2. Initialize logger
                self._logger_service = (
                    self._logger_service_provider()
                )
                self._logger_service.start()

                # 3. Initialize event bus
                self._event_bus = self._event_bus_provider()

                self._event_bus.publish(RuntimeStarting())

                # 4. Mark application as running
                self._state = RuntimeState.RUNNING

                self._logger_service.get_logger(
                    "RuntimeManager"
                ).success("Seven runtime started")

                self._event_bus.publish(RuntimeStarted())

                return True

            except Exception as exc:
                self._state = RuntimeState.FAILED

                if self._logger_service is not None:
                    if self._logger_service.is_running:
                        self._logger_service.get_logger(
                            "RuntimeManager"
                        ).exception("Seven startup failed")

                if self._event_bus is not None:
                    self._event_bus.publish(
                        RuntimeFailed(reason=str(exc))
                    )

                if (
                    self._logger_service is not None
                    and self._logger_service.is_running
                ):
                    self._logger_service.stop()

                raise

    def stop(self) -> bool:
        """Gracefully shut down Seven."""

        with self._lock:
            if self._state == RuntimeState.STOPPED:
                return False

            if self._state in (
                RuntimeState.STARTING,
                RuntimeState.STOPPING,
            ):
                raise RuntimeError(
                    f"Cannot stop while runtime is {self._state.value}"
                )

            self._state = RuntimeState.STOPPING

            try:
                # Notify subscribers before stopping the logger.
                if self._event_bus is not None:
                    self._event_bus.publish(RuntimeStopping())

                if (
                    self._logger_service is not None
                    and self._logger_service.is_running
                ):
                    self._logger_service.get_logger(
                        "RuntimeManager"
                    ).info("Seven runtime shutting down")

                self._state = RuntimeState.STOPPED

                if self._event_bus is not None:
                    self._event_bus.publish(RuntimeStopped())

                if (
                    self._logger_service is not None
                    and self._logger_service.is_running
                ):
                    self._logger_service.stop()

                self._logger_service = None
                self._event_bus = None

                return True

            except Exception:
                self._state = RuntimeState.FAILED

                if (
                    self._logger_service is not None
                    and self._logger_service.is_running
                ):
                    self._logger_service.get_logger(
                        "RuntimeManager"
                    ).exception("Seven shutdown failed")

                raise