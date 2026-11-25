
"""Coordinate transactional AI configuration updates."""

from __future__ import annotations

import asyncio

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIConfigurationError
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.ai.runtimes import LocalRuntimeFactory
from seven.core.config.config_manager import ConfigurationManager
from seven.core.config.config_schema import AIConfig


class AIConfigurationCoordinator:
    """Coordinate configuration reloads and provider replacement."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        engine: AIEngine,
        request_factory: GenerationRequestFactory,
        local_runtime_factory: LocalRuntimeFactory,
    ) -> None:
        if not isinstance(config_manager, ConfigurationManager):
            raise TypeError(
                "config_manager must be a ConfigurationManager."
            )

        if not isinstance(engine, AIEngine):
            raise TypeError("engine must be an AIEngine.")

        if not isinstance(
            request_factory,
            GenerationRequestFactory,
        ):
            raise TypeError(
                "request_factory must be a GenerationRequestFactory."
            )

        if not isinstance(
            local_runtime_factory,
            LocalRuntimeFactory,
        ):
            raise TypeError(
                "local_runtime_factory must be a LocalRuntimeFactory."
            )

        self._config_manager = config_manager
        self._engine = engine
        self._request_factory = request_factory
        self._local_runtime_factory = local_runtime_factory
        self._lock = asyncio.Lock()

    @property
    def config_manager(self) -> ConfigurationManager:
        return self._config_manager

    @property
    def engine(self) -> AIEngine:
        return self._engine

    @property
    def request_factory(self) -> GenerationRequestFactory:
        return self._request_factory

    @property
    def active_provider(self) -> AIProvider:
        return self._engine.provider

    async def reload(self) -> AIProvider:
        """Prepare and atomically activate updated AI configuration."""

        async with self._lock:
            if self._engine.is_closed:
                raise AIConfigurationError(
                    "Cannot reload AI configuration: AI engine is closed."
                )

            candidate = self._config_manager.prepare_reload()
            ai_config = candidate.config.ai

            if not isinstance(ai_config, AIConfig):
                raise AIConfigurationError(
                    "Loaded AI configuration has an invalid type."
                )

            new_provider = self._build_provider(ai_config)

            try:
                self._engine._validate_provider(new_provider)

                previous_request_config = self._request_factory.config

                def commit_configuration() -> None:
                    self._request_factory.update_config(ai_config)

                    try:
                        self._config_manager.commit_candidate(candidate)
                    except BaseException:
                        self._request_factory.update_config(
                            previous_request_config
                        )
                        raise

                await self._engine.switch_provider_instance(
                    new_provider,
                    on_switch=commit_configuration,
                )

            except BaseException:
                if new_provider is not self._engine.provider:
                    try:
                        await new_provider.close()
                    except Exception:
                        pass

                raise

            return new_provider

    def _build_provider(
        self,
        config: AIConfig,
    ) -> AIProvider:
        """Build a provider from a validated AI configuration."""

        mode = config.mode.strip().casefold()

        if mode == "local":
            try:
                return self._local_runtime_factory.create_provider(
                    config=config,
                    provider_name="local",
                )
            except (TypeError, ValueError) as exc:
                raise AIConfigurationError(
                    f"Invalid local AI configuration: {exc}"
                ) from exc

        if mode != "api":
            raise AIConfigurationError(
                f"Unsupported AI mode: {mode!r}."
            )

        provider_name = config.provider.strip().casefold()

        if config.api_key is None:
            raise AIConfigurationError(
                f"API key is missing for provider '{provider_name}'."
            )

        api_key = config.api_key.get_secret_value()

        if not api_key.strip():
            raise AIConfigurationError(
                f"API key is empty for provider '{provider_name}'."
            )

        if provider_name == "openai":
            try:
                return OpenAIProvider(
                    api_key=api_key,
                    model=config.model,
                    base_url=config.api_base_url,
                    timeout=config.request_timeout,
                )
            except (TypeError, ValueError) as exc:
                raise AIConfigurationError(
                    f"Invalid OpenAI configuration: {exc}"
                ) from exc

        if not config.api_base_url:
            raise AIConfigurationError(
                f"API base URL is missing for provider "
                f"'{provider_name}'."
            )

        try:
            return OpenAICompatibleProvider(
                api_key=api_key,
                provider_name=provider_name,
                base_url=config.api_base_url,
                model=config.model,
                timeout=config.request_timeout,
            )
        except (TypeError, ValueError) as exc:
            raise AIConfigurationError(
                f"Invalid configuration for provider "
                f"'{provider_name}': {exc}"
            ) from exc
