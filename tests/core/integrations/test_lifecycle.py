
import pytest

from seven.core.integrations import (
    IntegrationService,
    IntegrationServiceManager,
)


class FakeService:
    def __init__(
        self,
        name: str,
        events: list[str],
        fail_start: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self._name = name
        self._events = events
        self._running = False
        self.fail_start = fail_start
        self.fail_stop = fail_stop

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._events.append(f"start:{self.name}")

        if self.fail_start:
            raise RuntimeError(f"{self.name} startup failed")

        self._running = True

    def stop(self) -> None:
        self._events.append(f"stop:{self.name}")

        if self.fail_stop:
            raise RuntimeError(f"{self.name} shutdown failed")

        self._running = False


def test_manager_accepts_integration_services() -> None:
    events: list[str] = []
    service = FakeService("filesystem", events)

    manager = IntegrationServiceManager([service])

    assert isinstance(service, IntegrationService)
    assert manager.services == (service,)
    assert manager.is_running is False


def test_services_start_in_registration_order() -> None:
    events: list[str] = []

    services = [
        FakeService("filesystem", events),
        FakeService("processes", events),
        FakeService("applications", events),
    ]

    manager = IntegrationServiceManager(services)

    assert manager.start() is True
    assert events == [
        "start:filesystem",
        "start:processes",
        "start:applications",
    ]
    assert manager.is_running is True


def test_start_is_idempotent() -> None:
    events: list[str] = []
    manager = IntegrationServiceManager(
        [FakeService("filesystem", events)]
    )

    assert manager.start() is True
    assert manager.start() is False
    assert events == ["start:filesystem"]


def test_services_stop_in_reverse_order() -> None:
    events: list[str] = []

    services = [
        FakeService("filesystem", events),
        FakeService("processes", events),
        FakeService("applications", events),
    ]

    manager = IntegrationServiceManager(services)
    manager.start()

    assert manager.stop() is True
    assert events == [
        "start:filesystem",
        "start:processes",
        "start:applications",
        "stop:applications",
        "stop:processes",
        "stop:filesystem",
    ]
    assert manager.is_running is False


def test_stop_is_idempotent() -> None:
    events: list[str] = []
    manager = IntegrationServiceManager(
        [FakeService("filesystem", events)]
    )

    assert manager.stop() is False


def test_start_failure_rolls_back_started_services() -> None:
    events: list[str] = []

    services = [
        FakeService("filesystem", events),
        FakeService("processes", events, fail_start=True),
        FakeService("applications", events),
    ]

    manager = IntegrationServiceManager(services)

    with pytest.raises(RuntimeError, match="processes startup failed"):
        manager.start()

    assert events == [
        "start:filesystem",
        "start:processes",
        "stop:processes",
        "stop:filesystem",
    ]
    assert manager.is_running is False


def test_failed_shutdown_keeps_service_available_for_retry() -> None:
    events: list[str] = []
    service = FakeService(
        "filesystem",
        events,
        fail_stop=True,
    )

    manager = IntegrationServiceManager([service])
    manager.start()

    with pytest.raises(ExceptionGroup):
        manager.stop()

    assert manager.is_running is True
    assert manager.active_services == (service,)


def test_shutdown_can_be_retried() -> None:
    events: list[str] = []
    service = FakeService("filesystem", events)

    manager = IntegrationServiceManager([service])
    manager.start()

    assert manager.stop() is True
    assert manager.stop() is False


def test_register_rejects_duplicate_names() -> None:
    events: list[str] = []
    manager = IntegrationServiceManager(
        [FakeService("filesystem", events)]
    )

    with pytest.raises(ValueError, match="already registered"):
        manager.register(FakeService("filesystem", events))


def test_register_rejects_invalid_service() -> None:
    manager = IntegrationServiceManager()

    with pytest.raises(TypeError):
        manager.register(object())


def test_register_is_rejected_while_running() -> None:
    events: list[str] = []
    manager = IntegrationServiceManager(
        [FakeService("filesystem", events)]
    )
    manager.start()

    with pytest.raises(RuntimeError, match="while integrations are running"):
        manager.register(FakeService("processes", events))
