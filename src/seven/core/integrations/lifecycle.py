
from collections.abc import Iterable
from threading import RLock

from seven.core.integrations.interfaces import IntegrationService


class IntegrationServiceManager:
    """Manage the lifecycle of registered integration services."""

    def __init__(
        self,
        services: Iterable[IntegrationService] = (),
    ) -> None:
        self._services: list[IntegrationService] = []
        self._started: list[IntegrationService] = []
        self._lock = RLock()

        for service in services:
            self.register(service)

    @property
    def services(self) -> tuple[IntegrationService, ...]:
        return tuple(self._services)

    @property
    def is_running(self) -> bool:
        return bool(self._started)

    @property
    def active_services(self) -> tuple[IntegrationService, ...]:
        return tuple(self._started)

    def register(self, service: IntegrationService) -> None:
        """Register an integration service."""

        if not isinstance(service, IntegrationService):
            raise TypeError(
                "service must implement IntegrationService."
            )

        with self._lock:
            if self._started:
                raise RuntimeError(
                    "Cannot register services while integrations are running."
                )

            if any(item.name == service.name for item in self._services):
                raise ValueError(
                    f"Integration service '{service.name}' is already registered."
                )

            self._services.append(service)

    def start(self) -> bool:
        """Start all registered services in registration order."""

        with self._lock:
            if self._started:
                return False

            for service in self._services:
                try:
                    service.start()
                    self._started.append(service)

                except Exception as startup_error:
                    cleanup_errors: list[Exception] = []

                    rollback_services = [
                        service,
                        *reversed(self._started),
                    ]

                    for rollback_service in rollback_services:
                        try:
                            rollback_service.stop()
                        except Exception as cleanup_error:
                            cleanup_errors.append(cleanup_error)

                    self._started.clear()

                    if cleanup_errors:
                        raise ExceptionGroup(
                            "Integration startup and rollback failed.",
                            [startup_error, *cleanup_errors],
                        ) from startup_error

                    raise

            return True

    def stop(self) -> bool:
        """Stop active services in reverse registration order."""

        with self._lock:
            if not self._started:
                return False

            remaining: list[IntegrationService] = []
            errors: list[Exception] = []

            for service in reversed(self._started):
                try:
                    service.stop()
                except Exception as error:
                    errors.append(error)
                    remaining.append(service)

            self._started = list(reversed(remaining))

            if errors:
                raise ExceptionGroup(
                    "One or more integration services failed to stop.",
                    errors,
                )

            return True
