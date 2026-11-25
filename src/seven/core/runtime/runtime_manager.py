
from collections.abc import Callable
from threading import RLock

from seven.core.config import ConfigurationManager
from seven.core.events import EventBus
from seven.core.integrations import IntegrationServiceManager
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
    """Control the lifecycle of the Seven application."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        logger_service_provider: Callable[[], LoggerService],
        event_bus_provider: Callable[[], EventBus],
        integration_manager_provider: (
            Callable[[], IntegrationServiceManager] | None
        ) = None,
    ) -> None:
        self._config_manager = config_manager
        self._logger_service_provider = logger_service_provider
        self._event_bus_provider = event_bus_provider
        self._integration_manager_provider = integration_manager_provider

        self._logger_service: LoggerService | None = None
        self._event_bus: EventBus | None = None
        self._integration_manager: IntegrationServiceManager | None = None

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
                if not self._config_manager.is_loaded:
                    self._config_manager.load()

                self._logger_service = self._logger_service_provider()
                self._logger_service.start()

                self._event_bus = self._event_bus_provider()
                self._event_bus.publish(RuntimeStarting())

                if self._integration_manager_provider is not None:
                    self._integration_manager = (
                        self._integration_manager_provider()
                    )
                    self._integration_manager.start()

                self._state = RuntimeState.RUNNING

                self._logger_service.get_logger(
                    "RuntimeManager"
                ).success("Seven runtime started")

                self._event_bus.publish(RuntimeStarted())

                return True

            except Exception as exc:
                self._state = RuntimeState.FAILED

                if (
                    self._integration_manager is not None
                    and self._integration_manager.is_running
                ):
                    try:
                        self._integration_manager.stop()
                    except Exception:
                        if self._logger_service is not None:
                            self._logger_service.get_logger(
                                "RuntimeManager"
                            ).exception(
                                "Integration rollback failed"
                            )

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

                self._integration_manager = None
                self._logger_service = None
                self._event_bus = None

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
                if self._event_bus is not None:
                    self._event_bus.publish(RuntimeStopping())

                if (
                    self._logger_service is not None
                    and self._logger_service.is_running
                ):
                    self._logger_service.get_logger(
                        "RuntimeManager"
                    ).info("Seven runtime shutting down")

                if (
                    self._integration_manager is not None
                    and self._integration_manager.is_running
                ):
                    self._integration_manager.stop()

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
                self._integration_manager = None

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
