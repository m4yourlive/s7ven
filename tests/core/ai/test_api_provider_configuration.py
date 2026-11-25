
"""Tests for API provider configuration compatibility."""

from pydantic import SecretStr

from seven.core.config.config_schema import AIConfig


def test_ai_config_supports_openai_provider() -> None:
    config = AIConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=SecretStr("test-api-key"),
    )

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "test-api-key"


def test_ai_config_supports_openai_compatible_provider() -> None:
    config = AIConfig(
        provider="openrouter",
        model="test-model",
        api_key=SecretStr("test-api-key"),
        api_base_url="https://openrouter.ai/api/v1",
    )

    assert config.provider == "openrouter"
    assert config.model == "test-model"
    assert config.api_base_url == "https://openrouter.ai/api/v1"


def test_ai_config_supports_custom_api_endpoint() -> None:
    config = AIConfig(
        provider="custom-provider",
        model="custom-model",
        api_key=SecretStr("custom-api-key"),
        api_base_url="https://api.example.com/v1",
    )

    assert config.provider == "custom-provider"
    assert config.model == "custom-model"
    assert config.api_base_url == "https://api.example.com/v1"


def test_ai_config_preserves_existing_generation_defaults() -> None:
    config = AIConfig()

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.7
    assert config.top_p == 1.0
    assert config.request_timeout == 60.0
    assert config.stream_enabled is True
