
import pytest

from seven.core.config import (
    AppConfig,
    ConfigurationFileNotFoundError,
    ConfigurationManager,
    ConfigurationValidationError,
)


def create_config(tmp_path, content):
    config_file = tmp_path / "app.yaml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


def test_load_yaml_configuration(tmp_path):
    config_file = create_config(
        tmp_path,
        """
app:
  name: Seven
runtime:
  debug: true
ai:
  provider: openai
""",
    )

    manager = ConfigurationManager(
        config_path=config_file,
        schema=AppConfig,
    )

    config = manager.load()

    assert config.app.name == "Seven"
    assert config.runtime.debug is True
    assert config.ai.provider == "openai"


def test_environment_override(tmp_path, monkeypatch):
    config_file = create_config(
        tmp_path,
        """
ai:
  provider: openai
  model: default-model
""",
    )

    monkeypatch.setenv(
        "SEVEN_AI__PROVIDER",
        "gemini",
    )

    manager = ConfigurationManager(
        config_path=config_file,
        schema=AppConfig,
    )

    config = manager.load()

    assert config.ai.provider == "gemini"
    assert config.ai.model == "default-model"


def test_dot_notation_get(tmp_path):
    config_file = create_config(
        tmp_path,
        """
app:
  name: Seven
""",
    )

    manager = ConfigurationManager(
        config_path=config_file,
        schema=AppConfig,
    )

    manager.load()

    assert manager.get("app.name") == "Seven"
    assert manager.get("ai.provider") == "openai"
    assert manager.get("unknown.key", "fallback") == "fallback"


def test_invalid_configuration(tmp_path):
    config_file = create_config(
        tmp_path,
        """
ai:
  request_timeout: -10
""",
    )

    manager = ConfigurationManager(
        config_path=config_file,
        schema=AppConfig,
    )

    with pytest.raises(ConfigurationValidationError):
        manager.load()


def test_missing_file(tmp_path):
    manager = ConfigurationManager(
        config_path=tmp_path / "missing.yaml",
        schema=AppConfig,
    )

    with pytest.raises(ConfigurationFileNotFoundError):
        manager.load()


def test_reload_configuration(tmp_path):
    config_file = create_config(
        tmp_path,
        """
app:
  name: Seven
""",
    )

    manager = ConfigurationManager(
        config_path=config_file,
        schema=AppConfig,
    )

    manager.load()

    config_file.write_text(
        "app:\n  name: Seven Updated\n",
        encoding="utf-8",
    )

    updated = manager.reload()

    assert updated.app.name == "Seven Updated"
