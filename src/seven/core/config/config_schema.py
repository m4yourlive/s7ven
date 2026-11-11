from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

class StrictConfigModel(BaseModel):
    """Base model for Seven configuration"""

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
    api_base_url: str | None = None
    request_timeout: float = Field(default= 60.0, gt=0)

class VoiceConfig(StrictConfigModel):
    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    speech_rate: int = Field(default=175, ge=80, le=300)

class AppConfig(StrictConfigModel):
    app: AppIdentityConfig = Field(
        default_factory=AppIdentityConfig)
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig)
    storage: StorageConfig = Field(
        default_factory=StorageConfig)
    ai: AIConfig = Field(
        default_factory=AIConfig)
    voice: VoiceConfig = Field(
        default_factory=VoiceConfig)