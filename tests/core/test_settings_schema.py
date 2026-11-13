import pytest
from pydantic import ValidationError

from seven.core.settings import (
    AppearanceSettings,
    AudioSettings,
    UserSettings,
)


def test_user_settings_default():
    settings = UserSettings()

    assert settings.appearance.theme == "system"
    assert settings.appearance.accent_color == "#FF2D2D"
    assert settings.audio.master_volume == 80
    assert settings.audio.muted is False
    assert settings.general.language == "en"
    assert settings.general.notifications_enabled is True

def test_valid_theme():
    settings = AppearanceSettings(theme="dark")

    assert settings.theme == "dark"

def test_invalid_theme():
    with pytest.raises(ValidationError):
        AppearanceSettings(theme="neon")

def test_invalid_accent_color():
    with pytest.raises(ValidationError):
        AppearanceSettings(accent_color="red")

def test_volume_validation():
    AudioSettings(master_volume=100)

def test_assignment_validation():
    settings = UserSettings()

    with pytest.raises(ValidationError):
        settings.audio.master_volume = 150

def test_extra_fields_forbidden():
    settings = UserSettings()
    data = settings.model_dump()

    assert isinstance(data, dict)
    assert "appearance" in data
    assert "audio" in data
    assert "general" in data

