
from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from seven.core.events import EventBus
from seven.core.settings.defaults import create_default_settings
from seven.core.settings.events import (
    SettingsChanged,
    SettingsLoaded,
    SettingsReset,
    SettingsSaved,
)
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
        event_bus: EventBus | None = None,
    ) -> None:
        self._settings: UserSettings | None = None
        self._store = store
        self._event_bus = event_bus

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

        self._publish(SettingsLoaded())

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

        old_data = self.as_dict()
        data = deepcopy(old_data)

        path = key.split(".")

        self._set_nested(data, path, value)
        self._validate_and_apply(data)

        self._publish_changes(old_data, self.as_dict())

    def update(self, data: dict[str, Any]) -> None:
        """Update multiple settings with deep merge."""

        if not isinstance(data, dict):
            raise TypeError(
                "Settings update must be a dictionary."
            )

        old_data = self.as_dict()
        merged = self._deep_merge(old_data, data)

        self._validate_and_apply(merged)

        self._publish_changes(old_data, self.as_dict())

    def reset(self) -> UserSettings:
        """Reset all settings to their default values."""

        self._settings = create_default_settings()

        self._publish(SettingsReset())

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

        self._publish(SettingsSaved())

    def as_dict(self) -> dict[str, Any]:
        """Return a safe dictionary copy of current settings."""

        return self.settings.model_dump(mode="python")

    def _publish(self, event: Any) -> None:
        """Publish an event when Event Bus is available."""

        if self._event_bus is not None:
            self._event_bus.publish(event)

    def _publish_changes(
        self,
        old_data: dict[str, Any],
        new_data: dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Publish events for individual changed setting paths."""

        for key, old_value, new_value in self._collect_changes(
            old_data,
            new_data,
            prefix,
        ):
            self._publish(
                SettingsChanged(
                    key=key,
                    old_value=deepcopy(old_value),
                    new_value=deepcopy(new_value),
                )
            )

    @classmethod
    def _collect_changes(
        cls,
        old_data: dict[str, Any],
        new_data: dict[str, Any],
        prefix: str = "",
    ) -> list[tuple[str, Any, Any]]:
        """Collect changed values using dot notation."""

        changes: list[tuple[str, Any, Any]] = []

        keys = dict.fromkeys(
            (*old_data.keys(), *new_data.keys())
        )

        for key in keys:
            old_value = old_data.get(key)
            new_value = new_data.get(key)

            full_key = (
                f"{prefix}.{key}" if prefix else key
            )

            if (
                isinstance(old_value, dict)
                and isinstance(new_value, dict)
            ):
                changes.extend(
                    cls._collect_changes(
                        old_value,
                        new_value,
                        full_key,
                    )
                )

            elif old_value != new_value:
                changes.append(
                    (full_key, old_value, new_value)
                )

        return changes

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
