
"""Data contracts for user profiles and preferences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


ProfileValue = str | bool | int | float


def _normalize_key(value: str, field_name: str) -> str:
    """Normalize and validate a profile or preference key."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = value.strip().casefold()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")

    return normalized


def _normalize_value(value: ProfileValue, field_name: str) -> ProfileValue:
    """Validate a serializable scalar profile value."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")

        return normalized

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")

        return value

    raise TypeError(
        f"{field_name} must be a string, boolean, integer, or float."
    )


def _normalize_confidence(value: float) -> float:
    """Validate a normalized confidence value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Confidence must be numeric.")

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError("Confidence must be finite.")

    if not 0.0 <= normalized <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")

    return normalized


def _normalize_source(value: str) -> str:
    """Normalize and validate the origin of profile data."""
    if not isinstance(value, str):
        raise TypeError("Source must be a string.")

    normalized = value.strip().casefold()

    if not normalized:
        raise ValueError("Source cannot be empty.")

    return normalized


def _validate_timestamp(value: datetime | None, field_name: str) -> None:
    """Validate an optional timestamp."""
    if value is not None and not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime value.")


@dataclass(frozen=True, slots=True)
class UserProfileAttribute:
    """Represent one structured attribute in a user profile."""

    key: str
    value: ProfileValue
    confidence: float = 1.0
    source: str = "explicit"
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the profile attribute."""
        object.__setattr__(
            self,
            "key",
            _normalize_key(self.key, "Profile attribute key"),
        )
        object.__setattr__(
            self,
            "value",
            _normalize_value(self.value, "Profile attribute value"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_source(self.source),
        )
        _validate_timestamp(self.updated_at, "Updated at")


@dataclass(frozen=True, slots=True)
class UserPreference:
    """Represent one user preference."""

    key: str
    value: ProfileValue
    confidence: float = 1.0
    source: str = "explicit"
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the user preference."""
        object.__setattr__(
            self,
            "key",
            _normalize_key(self.key, "Preference key"),
        )
        object.__setattr__(
            self,
            "value",
            _normalize_value(self.value, "Preference value"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_source(self.source),
        )
        _validate_timestamp(self.updated_at, "Updated at")


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Represent a structured snapshot of a user's profile."""

    attributes: tuple[UserProfileAttribute, ...] = ()
    preferences: tuple[UserPreference, ...] = ()
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate profile collections and enforce unique keys."""
        if not isinstance(self.attributes, (tuple, list)):
            raise TypeError("Profile attributes must be a tuple or list.")

        if not isinstance(self.preferences, (tuple, list)):
            raise TypeError("Profile preferences must be a tuple or list.")

        attributes = tuple(self.attributes)
        preferences = tuple(self.preferences)

        if not all(
            isinstance(item, UserProfileAttribute)
            for item in attributes
        ):
            raise TypeError(
                "Profile attributes must contain UserProfileAttribute values."
            )

        if not all(
            isinstance(item, UserPreference)
            for item in preferences
        ):
            raise TypeError(
                "Profile preferences must contain UserPreference values."
            )

        attribute_keys = [item.key for item in attributes]
        preference_keys = [item.key for item in preferences]

        if len(attribute_keys) != len(set(attribute_keys)):
            raise ValueError("Profile attribute keys must be unique.")

        if len(preference_keys) != len(set(preference_keys)):
            raise ValueError("Preference keys must be unique.")

        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "preferences", preferences)

        _validate_timestamp(self.updated_at, "Profile updated at")


@dataclass(frozen=True, slots=True)
class ProfileUpdate:
    """Represent an update to one user profile attribute."""

    key: str
    value: ProfileValue
    confidence: float = 1.0
    source: str = "explicit"
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the profile update."""
        object.__setattr__(
            self,
            "key",
            _normalize_key(self.key, "Profile update key"),
        )
        object.__setattr__(
            self,
            "value",
            _normalize_value(self.value, "Profile update value"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_source(self.source),
        )
        _validate_timestamp(self.updated_at, "Updated at")

    def to_attribute(self) -> UserProfileAttribute:
        """Convert the update into a profile attribute."""
        return UserProfileAttribute(
            key=self.key,
            value=self.value,
            confidence=self.confidence,
            source=self.source,
            updated_at=self.updated_at,
        )


@dataclass(frozen=True, slots=True)
class PreferenceUpdate:
    """Represent an update to one user preference."""

    key: str
    value: ProfileValue
    confidence: float = 1.0
    source: str = "explicit"
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the preference update."""
        object.__setattr__(
            self,
            "key",
            _normalize_key(self.key, "Preference update key"),
        )
        object.__setattr__(
            self,
            "value",
            _normalize_value(self.value, "Preference update value"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_source(self.source),
        )
        _validate_timestamp(self.updated_at, "Updated at")

    def to_preference(self) -> UserPreference:
        """Convert the update into a user preference."""
        return UserPreference(
            key=self.key,
            value=self.value,
            confidence=self.confidence,
            source=self.source,
            updated_at=self.updated_at,
        )
