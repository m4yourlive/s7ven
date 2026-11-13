import pytest

from seven.core.settings.exceptions import (
    SettingsError,
    SettingsNotLoadedError,
    SettingsPersistenceError,
    SettingsValidationError,
)


def test_settings_error_is_base_exception():
    error = SettingsError("Settings error")

    assert isinstance(error, Exception)


@pytest.mark.parametrize(
    "exception_class",
    [
        SettingsNotLoadedError,
        SettingsValidationError,
        SettingsPersistenceError,
    ],
)
def test_settings_exceptions_inherit_from_base(exception_class):
    error = exception_class("Test error")

    assert isinstance(error, SettingsError)
    assert str(error) == "Test error"


def test_settings_not_loaded_error():
    with pytest.raises(SettingsNotLoadedError):
        raise SettingsNotLoadedError("Settings not loaded")


def test_settings_validation_error():
    with pytest.raises(SettingsValidationError):
        raise SettingsValidationError("Invalid settings")


def test_settings_persistence_error():
    with pytest.raises(SettingsPersistenceError):
        raise SettingsPersistenceError("Failed to save settings")
