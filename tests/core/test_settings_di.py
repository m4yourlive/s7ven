from seven.core.di import SevenContainer
from seven.core.settings.settings_manager import SettingsManager
from seven.core.settings.settings_store import SettingsStore


def create_test_container():
    container = SevenContainer()
    container.config_manager().load()
    return container


def test_settings_store_registered():
    container = create_test_container()

    store = container.settings_store()

    assert isinstance(store, SettingsStore)


def test_settings_manager_registered():
    container = create_test_container()

    manager = container.settings_manager()

    assert isinstance(manager, SettingsManager)


def test_settings_manager_singleton():
    container = create_test_container()

    first = container.settings_manager()
    second = container.settings_manager()

    assert first is second


def test_settings_store_singleton():
    container = create_test_container()

    first = container.settings_store()
    second = container.settings_store()

    assert first is second


def test_settings_manager_dependencies():
    container = create_test_container()

    manager = container.settings_manager()

    assert manager._store is container.settings_store()
    assert manager._event_bus is container.event_bus()


def test_settings_manager_can_load():
    container = create_test_container()

    manager = container.settings_manager()

    settings = manager.load()

    assert settings is not None
    assert manager.is_loaded
