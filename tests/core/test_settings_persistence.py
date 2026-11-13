import pytest

from seven.core.settings import (
    SettingsManager,
    SettingsNotLoadedError,
    SettingsPersistenceError,
    SettingsStore,
    SettingsValidationError,
)


def test_manager_saves_settings(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    manager = SettingsManager(store)

    manager.load()
    manager.set("appearance.theme", "dark")
    manager.save()

    assert store.load()["appearance"]["theme"] == "dark"


def test_manager_restores_saved_settings(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")

    first = SettingsManager(store)
    first.load()
    first.set("appearance.theme", "dark")
    first.set("audio.master_volume", 40)
    first.save()

    second = SettingsManager(store)
    second.load()

    assert second.get("appearance.theme") == "dark"
    assert second.get("audio.master_volume") == 40


def test_manager_uses_defaults_when_file_missing(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    manager = SettingsManager(store)

    manager.load()

    assert manager.get("appearance.theme") == "system"


def test_manager_save_before_load():
    manager = SettingsManager()

    with pytest.raises(SettingsNotLoadedError):
        manager.save()


def test_manager_without_store_cannot_save():
    manager = SettingsManager()
    manager.load()

    with pytest.raises(SettingsPersistenceError):
        manager.save()


def test_manager_rejects_invalid_persisted_settings(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.save({
        "appearance": {
            "theme": "invalid",
        }
    })

    manager = SettingsManager(store)

    with pytest.raises(SettingsValidationError):
        manager.load()


def test_manager_propagates_corrupt_file_error(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{invalid", encoding="utf-8")

    manager = SettingsManager(SettingsStore(path))

    with pytest.raises(SettingsPersistenceError):
        manager.load()


def test_reset_can_be_persisted(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    manager = SettingsManager(store)

    manager.load()
    manager.set("appearance.theme", "dark")
    manager.save()

    manager.reset()
    manager.save()

    restored = SettingsManager(store)
    restored.load()

    assert restored.get("appearance.theme") == "system"
