
"""Application configuration schemas for S7VEN."""

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
)

class StrictConfigModel(BaseModel):
    """Base model for Seven configuration."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

class AppIdentityConfig(StrictConfigModel):
    name: str = "Seven"
    version: str = "0.1.0a0"
    environment: str = "development"

class RuntimeConfig(StrictConfigModel):
    debug: bool = True
    log_level: str = "DEBUG"
    auto_start: bool = False
    minimize_to_tray: bool = True

class StorageConfig(StrictConfigModel):
    data_directory: Path = Path("data")
    database_name: str = "seven.db"

class AIConfig(StrictConfigModel):
    mode: Literal["api", "local"] = "api"

    # API provider configuration
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    api_base_url: str | None = None

    # Local runtime configuration
    local_runtime: str = "ollama"
    local_model: str = "llama3.2"
    local_model_path: Path | None = None
    local_base_url: str | None = "http://localhost:11434"
    local_request_timeout: float = Field(default=120.0, gt=0)

    # Shared generation configuration
    request_timeout: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)

    max_tokens: int | None = Field(default=None, gt=0)

    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
    )

    stream_enabled: bool = True

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip().lower()

    @field_validator(
        "provider",
        "model",
        "local_runtime",
        "local_model",
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be empty.")

        return value.strip()

    @field_validator("api_base_url", "local_base_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError("URL cannot be empty.")

        return normalized

class VoiceConfig(StrictConfigModel):
    """Configure voice interaction behavior."""

    enabled: bool = True
    input_enabled: bool = True
    output_enabled: bool = True

    speech_rate: int = Field(default=175, ge=80, le=300)
    speech_voice: str = "alloy"

    wake_word_enabled: bool = False
    wake_word: str = "hey seven"

    vad_enabled: bool = True
    vad_threshold: float = Field(default=500.0, gt=0)
    silence_duration_ms: int = Field(default=600, ge=100, le=5000)

    max_audio_size_mb: int = Field(default=25, ge=1, le=100)

class AppConfig(StrictConfigModel):
    app: AppIdentityConfig = Field(
        default_factory=AppIdentityConfig,
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
    )
    ai: AIConfig = Field(
        default_factory=AIConfig,
    )
    voice: VoiceConfig = Field(
        default_factory=VoiceConfig,
    )