from seven.core.settings import UserSettings
from seven.core.settings.defaults import create_default_settings


def test_create_default_settings_returns_user_settings():
    settings = create_default_settings()

    assert isinstance(settings, UserSettings)


def test_default_settings_values():
    settings = create_default_settings()

    assert settings.appearance.theme == "system"
    assert settings.appearance.accent_color == "#FF2D2D"

    assert settings.audio.master_volume == 80
    assert settings.audio.muted is False

    assert settings.general.language == "en"
    assert settings.general.notifications_enabled is True


def test_default_settings_are_independent():
    first = create_default_settings()
    second = create_default_settings()

    first.appearance.theme = "dark"

    assert second.appearance.theme == "system"


def test_nested_settings_are_independent():
    first = create_default_settings()
    second = create_default_settings()

    first.audio.master_volume = 20

    assert second.audio.master_volume == 80
