
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from seven.core.config.exceptions import (
    ConfigurationFileNotFoundError,
    ConfigurationLoadError,
    ConfigurationValidationError,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ConfigurationCandidate(Generic[T]):
    """Validated configuration that has not yet been activated."""

    config: T
    raw_config: dict[str, Any]


class ConfigurationManager:
    """Centralized configuration manager for Seven."""

    def __init__(
        self,
        config_path: str | Path,
        schema: type[T],
        env_prefix: str = "SEVEN_",
        env_nested_delimiter: str = "__",
        dotenv_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.schema = schema
        self.env_prefix = env_prefix
        self.env_nested_delimiter = env_nested_delimiter
        self.dotenv_path = (
            Path(dotenv_path) if dotenv_path else None
        )

        self._config: T | None = None
        self._raw_config: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._config is not None

    @property
    def config(self) -> T:
        if self._config is None:
            raise ConfigurationLoadError(
                "Configuration has not been loaded. "
                "Call load() first."
            )

        return self._config

    def prepare_reload(self) -> ConfigurationCandidate[T]:
        """Load and validate a candidate without activating it."""
        yaml_data = self._load_yaml()

        if self.dotenv_path and self.dotenv_path.exists():
            load_dotenv(
                dotenv_path=self.dotenv_path,
                override=False,
            )

        env_data = self._load_environment()
        merged_data = self._deep_merge(yaml_data, env_data)

        try:
            validated = self.schema.model_validate(merged_data)
        except ValidationError as exc:
            raise ConfigurationValidationError(
                f"Invalid configuration: {exc}"
            ) from exc

        return ConfigurationCandidate(
            config=validated,
            raw_config=merged_data,
        )

    def commit_candidate(
        self,
        candidate: ConfigurationCandidate[T],
    ) -> T:
        """Activate a previously validated configuration candidate."""
        if not isinstance(candidate, ConfigurationCandidate):
            raise TypeError(
                "candidate must be a ConfigurationCandidate instance."
            )

        if not isinstance(candidate.config, self.schema):
            raise TypeError(
                "Candidate configuration does not match the schema."
            )

        self._raw_config = deepcopy(candidate.raw_config)
        self._config = candidate.config

        return self._config

    def load(self) -> T:
        """Load and activate configuration."""
        candidate = self.prepare_reload()
        return self.commit_candidate(candidate)

    def reload(self) -> T:
        """Reload configuration without discarding the active version."""
        return self.load()

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a configuration value using dot notation."""
        current: Any = self.config

        for part in key.split("."):
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

        return current

    def as_dict(self) -> dict[str, Any]:
        return self.config.model_dump(mode="python")

    def _load_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise ConfigurationFileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        try:
            with self.config_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = yaml.safe_load(file)

            if data is None:
                return {}

            if not isinstance(data, dict):
                raise ConfigurationLoadError(
                    "YAML root must be a mapping."
                )

            return data

        except yaml.YAMLError as exc:
            raise ConfigurationLoadError(
                f"Failed to parse YAML: {exc}"
            ) from exc

        except OSError as exc:
            raise ConfigurationLoadError(
                f"Failed to read configuration: {exc}"
            ) from exc

    def _load_environment(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue

            path = key[len(self.env_prefix):].lower().split(
                self.env_nested_delimiter
            )

            if not path or not all(path):
                continue

            try:
                parsed_value = yaml.safe_load(value)
            except yaml.YAMLError:
                parsed_value = value

            self._set_nested(result, path, parsed_value)

        return result

    @staticmethod
    def _set_nested(
        target: dict[str, Any],
        path: list[str],
        value: Any,
    ) -> None:
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
