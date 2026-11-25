
"""Tests for local AI configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from seven.core.config.config_schema import (
    AIConfig,
    AppConfig,
)

def test_ai_config_defaults_to_api_mode() -> None:
    config = AIConfig()

    assert config.mode == "api"
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"

def test_ai_config_has_local_runtime_defaults() -> None:
    config = AIConfig()

    assert config.local_runtime == "ollama"
    assert config.local_model == "llama3.2"
    assert config.local_model_path is None
    assert config.local_base_url == "http://localhost:11434"
    assert config.local_request_timeout == 120.0

def test_ai_config_accepts_local_mode() -> None:
    config = AIConfig(mode="local")

    assert config.mode == "local"

def test_ai_config_normalizes_mode() -> None:
    config = AIConfig(mode=" LOCAL ")

    assert config.mode == "local"

def test_ai_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValidationError):
        AIConfig(mode="hybrid")

def test_ai_config_accepts_custom_local_runtime() -> None:
    config = AIConfig(
        mode="local",
        local_runtime="llama.cpp",
        local_model="custom-model",
    )

    assert config.local_runtime == "llama.cpp"
    assert config.local_model == "custom-model"

def test_ai_config_accepts_local_model_path() -> None:
    model_path = Path("models/test-model.gguf")

    config = AIConfig(
        mode="local",
        local_model_path=model_path,
    )

    assert config.local_model_path == model_path

def test_ai_config_accepts_custom_local_endpoint() -> None:
    config = AIConfig(
        mode="local",
        local_base_url="http://127.0.0.1:8080",
    )

    assert config.local_base_url == "http://127.0.0.1:8080"

def test_ai_config_accepts_none_local_endpoint() -> None:
    config = AIConfig(local_base_url=None)

    assert config.local_base_url is None

@pytest.mark.parametrize(
    "field",
    [
        "local_runtime",
        "local_model",
    ],
)
def test_ai_config_rejects_empty_local_values(field: str) -> None:
    with pytest.raises(ValidationError):
        AIConfig(**{field: "  "})

def test_ai_config_rejects_empty_local_endpoint() -> None:
    with pytest.raises(ValidationError):
        AIConfig(local_base_url="  ")

def test_ai_config_rejects_non_positive_local_timeout() -> None:
    with pytest.raises(ValidationError):
        AIConfig(local_request_timeout=0)

def test_ai_config_preserves_api_configuration() -> None:
    config = AIConfig(
        mode="api",
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_base_url="https://openrouter.ai/api/v1",
        request_timeout=45.0,
    )

    assert config.mode == "api"
    assert config.provider == "openrouter"
    assert config.model == "openai/gpt-4o-mini"
    assert config.api_base_url == "https://openrouter.ai/api/v1"
    assert config.request_timeout == 45.0

def test_app_config_includes_local_ai_configuration() -> None:
    config = AppConfig(
        ai=AIConfig(
            mode="local",
            local_runtime="ollama",
            local_model="llama3.2",
        )
    )

    assert config.ai.mode == "local"
    assert config.ai.local_runtime == "ollama"
    assert config.ai.local_model == "llama3.2"

def test_ai_config_assignment_validates_mode() -> None:
    config = AIConfig()

    with pytest.raises(ValidationError):
        config.mode = "invalid"