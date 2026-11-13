from seven.core.settings.settings_schema import UserSettings


def create_default_settings() -> UserSettings:
    """Create a fresh instance of default user settings."""

    return UserSettings()
