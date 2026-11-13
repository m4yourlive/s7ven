from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from seven.core.events import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingsLoaded(Event):
    """Emitted when settings are successfully loaded."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingsChanged(Event):
    """Emitted when a setting value changes."""

    key: str
    old_value: Any
    new_value: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingsReset(Event):
    """Emitted when settings are successfully reset."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingsSaved(Event):
    """Emitted when settings are successfully saved."""
