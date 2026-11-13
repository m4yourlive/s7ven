import pytest

from seven.core.settings import (
    SettingsManager,
    SettingsNotLoadedError,
    SettingsValidationError,
    UserSettings,
)


def test_manager_initial_state():
    manager = SettingsManager()

    assert manager.is_loaded is False


def test_access_before_loading():
    manager = SettingsManager()

    with pytest.raises(SettingsNotLoadedError):
        _ = manager.settings


def test_load_default_settings():
    manager = SettingsManager()

    settings = manager.load()

    assert isinstance(settings, UserSettings)
    assert manager.is_loaded is True
    assert manager.get("appearance.theme") == "system"


def test_load_custom_settings():
    manager = SettingsManager()

    manager.load({
        "appearance": {
            "theme": "dark",
            "accent_color": "#FF0000",
        }
    })

    assert manager.get("appearance.theme") == "dark"
    assert manager.get("appearance.accent_color") == "#FF0000"


def test_load_invalid_settings():
    manager = SettingsManager()

    with pytest.raises(SettingsValidationError):
        manager.load({
            "appearance": {
                "theme": "invalid",
            }
        })

    assert manager.is_loaded is False


def test_get_with_default():
    manager = SettingsManager()
    manager.load()

    assert manager.get("unknown.key", "fallback") == "fallback"


def test_get_invalid_path():
    manager = SettingsManager()
    manager.load()

    assert manager.get("appearance.unknown") is None


def test_set_single_setting():
    manager = SettingsManager()
    manager.load()

    manager.set("appearance.theme", "dark")

    assert manager.get("appearance.theme") == "dark"


def test_set_invalid_setting():
    manager = SettingsManager()
    manager.load()

    with pytest.raises(SettingsValidationError):
        manager.set("appearance.theme", "neon")

    assert manager.get("appearance.theme") == "system"


def test_set_unknown_setting():
    manager = SettingsManager()
    manager.load()

    with pytest.raises(SettingsValidationError):
        manager.set("appearance.unknown", True)

    assert manager.get("appearance.theme") == "system"


def test_set_invalid_path():
    manager = SettingsManager()
    manager.load()

    with pytest.raises(ValueError):
        manager.set("appearance..theme", "dark")


def test_update_multiple_settings():
    manager = SettingsManager()
    manager.load()

    manager.update({
        "appearance": {
            "theme": "dark",
        },
        "audio": {
            "master_volume": 50,
        },
    })

    assert manager.get("appearance.theme") == "dark"
    assert manager.get("appearance.accent_color") == "#FF2D2D"
    assert manager.get("audio.master_volume") == 50


def test_update_invalid_settings_is_atomic():
    manager = SettingsManager()
    manager.load()

    with pytest.raises(SettingsValidationError):
        manager.update({
            "appearance": {"theme": "neon"},
            "audio": {"master_volume": 20},
        })

    assert manager.get("appearance.theme") == "system"
    assert manager.get("audio.master_volume") == 80


def test_update_requires_dictionary():
    manager = SettingsManager()
    manager.load()

    with pytest.raises(TypeError):
        manager.update("invalid")


def test_reset_settings():
    manager = SettingsManager()
    manager.load()

    manager.set("appearance.theme", "dark")
    manager.set("audio.master_volume", 20)

    settings = manager.reset()

    assert settings.appearance.theme == "system"
    assert settings.audio.master_volume == 80


def test_settings_returns_independent_copy():
    manager = SettingsManager()
    manager.load()

    external_settings = manager.settings
    external_settings.appearance.theme = "dark"

    assert manager.get("appearance.theme") == "system"


def test_as_dict_returns_independent_copy():
    manager = SettingsManager()
    manager.load()

    data = manager.as_dict()
    data["appearance"]["theme"] = "dark"

    assert manager.get("appearance.theme") == "system"


def test_load_existing_settings_model():
    manager = SettingsManager()
    original = UserSettings()

    manager.load(original)

    original.appearance.theme = "dark"

    assert manager.get("appearance.theme") == "system"
