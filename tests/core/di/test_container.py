"""Tests for the Seven dependency injection container."""

from unittest.mock import Mock

from dependency_injector import providers

from seven.core.di.container import (
    SevenContainer,
    initialize_container,
    shutdown_container,
)


def test_config_manager_is_singleton() -> None:
    container = SevenContainer()

    first = container.config_manager()
    second = container.config_manager()

    assert first is second


def test_runtime_manager_is_singleton() -> None:
    container = SevenContainer()

    first = container.runtime_manager()
    second = container.runtime_manager()

    assert first is second


def test_runtime_manager_receives_config_manager() -> None:
    container = SevenContainer()

    runtime_manager = container.runtime_manager()
    config_manager = container.config_manager()

    assert runtime_manager._config_manager is config_manager


def test_initialize_container_starts_runtime() -> None:
    container = SevenContainer()
    runtime_manager = Mock()

    with container.runtime_manager.override(
        providers.Object(runtime_manager)
    ):
        initialize_container(container)

    runtime_manager.start.assert_called_once_with()


def test_shutdown_container_stops_runtime() -> None:
    container = SevenContainer()
    runtime_manager = Mock()

    with container.runtime_manager.override(
        providers.Object(runtime_manager)
    ):
        shutdown_container(container)

    runtime_manager.stop.assert_called_once_with()


def test_initialize_and_shutdown_use_same_runtime() -> None:
    container = SevenContainer()
    runtime_manager = Mock()

    with container.runtime_manager.override(
        providers.Object(runtime_manager)
    ):
        initialize_container(container)
        shutdown_container(container)

    runtime_manager.start.assert_called_once_with()
    runtime_manager.stop.assert_called_once_with()
