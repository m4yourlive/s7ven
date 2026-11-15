
"""Application configuration schemas for S7VEN."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


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
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    api_base_url: str | None = None

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

    @field_validator("provider", "model")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be empty.")
        return value.strip()


class VoiceConfig(StrictConfigModel):
    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    speech_rate: int = Field(default=175, ge=80, le=300)


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
