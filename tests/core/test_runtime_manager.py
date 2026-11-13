from seven.core.events import EventBus
from seven.core.runtime import (
    RuntimeManager,
    RuntimeStarted,
    RuntimeStarting,
    RuntimeState,
    RuntimeStopped,
    RuntimeStopping,
)


class FakeConfigManager:
    def __init__(self, fail: bool = False):
        self.is_loaded = False
        self.fail = fail

    def load(self):
        if self.fail:
            raise RuntimeError("Configuration failed")

        self.is_loaded = True

class FakeLogger:
    def __init__(self, fail: bool = False):
        self.is_running = False
        self.fail = fail

    def start(self):
        if self.fail:
            raise RuntimeError("Logger failed")

        self.is_running = True

    def stop(self):
        self.is_running = False

    def get_logger(self, module: str):
        return FakeLog()

class FakeLog:
    def success(self, message: str):
        pass

    def info(self, message: str):
        pass

    def exception(self, message: str):
        pass

def create_manager(
    config=None,
    logger=None,
    bus=None,
):
    config = config or FakeConfigManager()
    logger = logger or FakeLogger()
    bus = bus or EventBus()

    manager = RuntimeManager(
        config_manager=config,
        logger_service_provider=lambda: logger,
        event_bus_provider=lambda: bus,
    )

    return manager, config, logger, bus

def test_initial_state():
    manager, _, _, _ = create_manager()

    assert manager.state == RuntimeState.STOPPED
    assert manager.is_running is False

def test_runtime_start():
    manager, config, logger, _ = create_manager()

    result = manager.start()

    assert result is True
    assert config.is_loaded
    assert logger.is_running
    assert manager.state == RuntimeState.RUNNING
    assert manager.is_running

def test_start_publishes_events():
    manager, _, _, bus = create_manager()
    received = []

    bus.subscribe(
        RuntimeStarting,
        lambda event: received.append("starting"),
    )

    bus.subscribe(
        RuntimeStarted,
        lambda event: received.append("started"),
    )

    manager.start()

    assert received == ["starting", "started"]

def test_start_is_idempotent():
    manager, _, logger, _ = create_manager()

    assert manager.start() is True
    assert manager.start() is False

    assert logger.is_running
    assert manager.state == RuntimeState.RUNNING

def test_runtime_stop():
    manager, _, logger, bus = create_manager()
    received = []

    manager.start()

    bus.subscribe(
        RuntimeStopping,
        lambda event: received.append("stopping"),
    )

    bus.subscribe(
        RuntimeStopped,
        lambda event: received.append("stopped"),
    )

    result = manager.stop()

    assert result is True
    assert received == ["stopping", "stopped"]
    assert logger.is_running is False
    assert manager.state == RuntimeState.STOPPED

def test_stop_is_idempotent():
    manager, _, _, _ = create_manager()

    manager.start()

    assert manager.stop() is True
    assert manager.stop() is False

    assert manager.state == RuntimeState.STOPPED

def test_config_failure():
    config = FakeConfigManager(fail=True)
    manager, _, logger, _ = create_manager(config=config)

    try:
        manager.start()
        assert False, "Expected startup failure"
    except RuntimeError as exc:
        assert str(exc) == "Configuration failed"

    assert manager.state == RuntimeState.FAILED
    assert logger.is_running is False

def test_logger_failure():
    logger = FakeLogger(fail=True)
    manager, _, _, _ = create_manager(logger=logger)

    try:
        manager.start()
        assert False, "Expected startup failure"
    except RuntimeError as exc:
        assert str(exc) == "Logger failed"

    assert manager.state == RuntimeState.FAILED
    assert logger.is_running is False