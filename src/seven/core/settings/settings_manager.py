from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from seven.core.settings.defaults import create_default_settings
from seven.core.settings.exceptions import (
    SettingsNotLoadedError,
    SettingsPersistenceError,
    SettingsValidationError,
)
from seven.core.settings.settings_schema import UserSettings
from seven.core.settings.settings_store import SettingsStore


class SettingsManager:
    """Centralized manager for Seven user settings."""

    def __init__(
        self,
        store: SettingsStore | None = None,
    ) -> None:
        self._settings: UserSettings | None = None
        self._store = store

    @property
    def is_loaded(self) -> bool:
        """Return whether settings have been loaded."""
        return self._settings is not None

    @property
    def settings(self) -> UserSettings:
        """Return a safe copy of current settings."""

        if self._settings is None:
            raise SettingsNotLoadedError(
                "Settings have not been loaded. Call load() first."
            )

        return self._settings.model_copy(deep=True)

    def load(
        self,
        settings: UserSettings | dict[str, Any] | None = None,
    ) -> UserSettings:
        """Load and validate settings."""

        try:
            if settings is None:
                if self._store is not None and self._store.exists:
                    settings = self._store.load()
                    validated = UserSettings.model_validate(settings)
                else:
                    validated = create_default_settings()

            elif isinstance(settings, UserSettings):
                validated = settings.model_copy(deep=True)

            else:
                validated = UserSettings.model_validate(settings)

        except ValidationError as exc:
            raise SettingsValidationError(
                f"Invalid settings data: {exc}"
            ) from exc

        self._settings = validated

        return self.settings

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a setting using dot notation."""

        if not isinstance(key, str) or not key:
            return default

        current: Any = self.settings

        for part in key.split("."):
            if not part:
                return default

            if isinstance(current, BaseModel):
                if part not in type(current).model_fields:
                    return default

                current = getattr(current, part)

            elif isinstance(current, dict):
                if part not in current:
                    return default

                current = current[part]

            else:
                return default

        return deepcopy(current)

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a single setting using dot notation."""

        if (
            not isinstance(key, str)
            or not key
            or any(not part for part in key.split("."))
        ):
            raise ValueError(
                "Setting key must be a valid dot path."
            )

        data = self.as_dict()
        path = key.split(".")

        self._set_nested(data, path, value)
        self._validate_and_apply(data)

    def update(self, data: dict[str, Any]) -> None:
        """Update multiple settings with deep merge."""

        if not isinstance(data, dict):
            raise TypeError(
                "Settings update must be a dictionary."
            )

        current = self.as_dict()
        merged = self._deep_merge(current, data)

        self._validate_and_apply(merged)

    def reset(self) -> UserSettings:
        """Reset all settings to their default values."""

        self._settings = create_default_settings()

        return self.settings

    def save(self) -> None:
        """Persist current settings to storage."""

        if self._settings is None:
            raise SettingsNotLoadedError(
                "Settings have not been loaded. Call load() first."
            )

        if self._store is None:
            raise SettingsPersistenceError(
                "No settings storage has been configured."
            )

        self._store.save(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        """Return a safe dictionary copy of current settings."""

        return self.settings.model_dump(mode="python")

    def _validate_and_apply(
        self,
        data: dict[str, Any],
    ) -> None:
        """Validate candidate settings before applying changes."""

        try:
            validated = UserSettings.model_validate(data)

        except ValidationError as exc:
            raise SettingsValidationError(
                f"Invalid settings update: {exc}"
            ) from exc

        self._settings = validated

    @staticmethod
    def _set_nested(
        target: dict[str, Any],
        path: list[str],
        value: Any,
    ) -> None:
        """Set a nested dictionary value."""

        current = target

        for part in path[:-1]:
            existing = current.get(part)

            if not isinstance(existing, dict):
                current[part] = {}

            current = current[part]

        current[path[-1]] = value

    @classmethod
    def _deep_merge(
        cls,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge settings dictionaries."""

        result = deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._deep_merge(
                    result[key],
                    value,
                )

            else:
                result[key] = deepcopy(value)

        return result
