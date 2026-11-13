from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSettingsModel(BaseModel):
    """Base model for Seven user settings."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

class AppearanceSettings(StrictSettingsModel):
    """User interface appearance preferences."""

    theme: Literal["system", "light", "dark"] = "system"

    accent_color: str = Field(
        default="#FF2D2D",
        pattern= r"^#[0-9A-Fa-f]{6}$",
    )

class AudioSettings(StrictSettingsModel):
    """Audio preferences."""

    master_volume: int = Field(
        default=80,
        ge=0,
        le=100,
    )

    muted: bool = False

class GeneralSettings(StrictSettingsModel):
    """General user preferences."""

    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
    )

    notifications_enabled: bool = True

class UserSettings(StrictSettingsModel):
    """Root model for Seven user settings."""

    appearance: AppearanceSettings = Field(
        default_factory=AppearanceSettings,
    )
    audio: AudioSettings = Field(
        default_factory=AudioSettings,
    )

    general: GeneralSettings = Field(
        default_factory=GeneralSettings,
    )