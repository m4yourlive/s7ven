class SettingsError(Exception):
    """Base exception for settings errors."""


class SettingsNotLoadedError(SettingsError):
    """Raised when settings have not been loaded."""


class SettingsValidationError(SettingsError):
    """Raised when settings data fails validation."""


class SettingsPersistenceError(SettingsError):
    """Raised when settings cannot be read or saved."""