from seven.core.events import EventBus
from seven.core.settings.events import (
    SettingsLoaded,
    SettingsReset,
)
from seven.core.settings.settings_manager import SettingsManager


def test_settings_loaded_event():
    event_bus = EventBus()
    received = []

    event_bus.subscribe(
        SettingsLoaded,
        received.append,
    )

    manager = SettingsManager(event_bus=event_bus)
    manager.load()

    assert len(received) == 1
    assert isinstance(received[0], SettingsLoaded)


def test_settings_reset_event():
    event_bus = EventBus()
    received = []

    event_bus.subscribe(
        SettingsReset,
        received.append,
    )

    manager = SettingsManager(event_bus=event_bus)
    manager.load()
    manager.reset()

    assert len(received) == 1
    assert isinstance(received[0], SettingsReset)


def test_settings_manager_without_event_bus():
    manager = SettingsManager()

    manager.load()

    assert manager.is_loaded
