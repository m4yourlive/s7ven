from __future__ import annotations

from types import SimpleNamespace

from dependency_injector import containers, providers

from seven.core.di.container import (
    SevenContainer,
    create_event_logger,
    create_logger_config,
    initialize_container,
    shutdown_container,
)
from seven.core.logging import LoggerConfig


class FakeLoggerService:
    """Fake logger service."""

    def __init__(self) -> None:
        self.requested_loggers: list[str] = []
        self.logger = object()

    def get_logger(self, name: str) -> object:
        self.requested_loggers.append(name)
        return self.logger


class FakeRuntimeManager:
    """Fake runtime manager."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class FakeContainer:
    """Fake dependency injection container."""

    def __init__(self) -> None:
        self.runtime = FakeRuntimeManager()

    def runtime_manager(self) -> FakeRuntimeManager:
        return self.runtime


def test_seven_container_can_be_initialized() -> None:
    container = SevenContainer()

    assert container is not None
    assert isinstance(container, containers.DynamicContainer)

    assert callable(container.runtime_manager)
    assert callable(container.config_manager)
    assert callable(container.logger_service)
    assert callable(container.event_bus)
    assert callable(container.settings_store)
    assert callable(container.settings_manager)


def test_container_defines_singleton_providers() -> None:
    container = SevenContainer()

    expected_providers = {
        "config_manager",
        "logger_service",
        "event_bus",
        "settings_store",
        "settings_manager",
        "runtime_manager",
    }

    assert expected_providers.issubset(container.providers)

    for provider_name in expected_providers:
        provider = container.providers[provider_name]

        assert isinstance(
            provider,
            providers.Singleton,
        ), f"{provider_name} must use Singleton provider"


def test_create_logger_config_uses_runtime_log_level() -> None:
    expected_level = "DEBUG"

    config_manager = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                log_level=expected_level,
            )
        )
    )

    result = create_logger_config(config_manager)

    assert isinstance(result, LoggerConfig)
    assert result.level == expected_level


def test_create_event_logger_uses_event_bus_name() -> None:
    logger_service = FakeLoggerService()

    result = create_event_logger(logger_service)

    assert result is logger_service.logger
    assert logger_service.requested_loggers == ["EventBus"]


def test_initialize_container_starts_runtime() -> None:
    container = FakeContainer()

    initialize_container(container)

    assert container.runtime.start_calls == 1
    assert container.runtime.stop_calls == 0


def test_shutdown_container_stops_runtime() -> None:
    container = FakeContainer()

    shutdown_container(container)

    assert container.runtime.stop_calls == 1
    assert container.runtime.start_calls == 0


def test_initialize_and_shutdown_use_same_runtime_instance() -> None:
    container = FakeContainer()

    initialize_container(container)
    shutdown_container(container)

    assert container.runtime_manager() is container.runtime
    assert container.runtime.start_calls == 1
    assert container.runtime.stop_calls == 1
