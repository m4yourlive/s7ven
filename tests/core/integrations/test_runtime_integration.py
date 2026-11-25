
from seven.core.events import EventBus
from seven.core.integrations import IntegrationServiceManager
from seven.core.runtime import RuntimeManager, RuntimeState


class FakeConfigManager:
    def __init__(self) -> None:
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True


class FakeLogger:
    def __init__(self) -> None:
        self.is_running = False

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def get_logger(self, name: str):
        return self

    def success(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def exception(self, message: str) -> None:
        pass


class FakeIntegrationService:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._running = False

    @property
    def name(self) -> str:
        return "fake"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.events.append("integration_started")
        self._running = True

    def stop(self) -> None:
        self.events.append("integration_stopped")
        self._running = False


def create_runtime():
    events: list[str] = []
    config = FakeConfigManager()
    logger = FakeLogger()
    event_bus = EventBus()

    integration_manager = IntegrationServiceManager(
        [FakeIntegrationService(events)]
    )

    runtime = RuntimeManager(
        config_manager=config,
        logger_service_provider=lambda: logger,
        event_bus_provider=lambda: event_bus,
        integration_manager_provider=lambda: integration_manager,
    )

    return runtime, logger, integration_manager, events


def test_runtime_starts_integration_services() -> None:
    runtime, _, integration_manager, events = create_runtime()

    assert runtime.start() is True

    assert runtime.state == RuntimeState.RUNNING
    assert integration_manager.is_running is True
    assert events == ["integration_started"]


def test_runtime_stops_integration_services() -> None:
    runtime, _, integration_manager, events = create_runtime()

    runtime.start()

    assert runtime.stop() is True

    assert runtime.state == RuntimeState.STOPPED
    assert integration_manager.is_running is False
    assert events == [
        "integration_started",
        "integration_stopped",
    ]


def test_runtime_start_remains_idempotent() -> None:
    runtime, _, _, events = create_runtime()

    assert runtime.start() is True
    assert runtime.start() is False

    assert events == ["integration_started"]


def test_runtime_shutdown_remains_idempotent() -> None:
    runtime, _, _, events = create_runtime()

    runtime.start()

    assert runtime.stop() is True
    assert runtime.stop() is False

    assert events == [
        "integration_started",
        "integration_stopped",
    ]


def test_runtime_without_integration_provider_remains_compatible() -> None:
    config = FakeConfigManager()
    logger = FakeLogger()
    event_bus = EventBus()

    runtime = RuntimeManager(
        config_manager=config,
        logger_service_provider=lambda: logger,
        event_bus_provider=lambda: event_bus,
    )

    assert runtime.start() is True
    assert runtime.state == RuntimeState.RUNNING
    assert runtime.stop() is True
    assert runtime.state == RuntimeState.STOPPED
