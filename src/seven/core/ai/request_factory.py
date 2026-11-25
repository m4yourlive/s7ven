
"""Factory for building AI generation requests from configuration and prompts."""

from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.prompts.prompt_manager import PromptManager
from seven.core.config.config_schema import AIConfig


class GenerationRequestFactory:
    """Build generation requests with the configured system prompt."""

    def __init__(
        self,
        config: AIConfig,
        prompt_manager: PromptManager,
    ) -> None:
        if not isinstance(config, AIConfig):
            raise TypeError("config must be an AIConfig instance.")

        if not isinstance(prompt_manager, PromptManager):
            raise TypeError(
                "prompt_manager must be a PromptManager instance."
            )

        self._config = config
        self._prompt_manager = prompt_manager

    @property
    def config(self) -> AIConfig:
        """Return the currently active AI configuration."""
        return self._config

    def update_config(self, config: AIConfig) -> None:
        """Replace the active AI configuration."""
        if not isinstance(config, AIConfig):
            raise TypeError("config must be an AIConfig instance.")

        self._config = config

    def build(
        self,
        messages: tuple[AIMessage, ...],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> GenerationRequest:
        """Build a generation request using the active AI configuration."""

        default_model = (
            self._config.local_model
            if self._config.mode == "local"
            else self._config.model
        )

        selected_model = default_model if model is None else model

        system_message = AIMessage(
            role=MessageRole.SYSTEM,
            content=self._prompt_manager.render("seven_system"),
        )

        request_messages = (system_message, *messages)

        return GenerationRequest(
            messages=request_messages,
            model=selected_model,
            temperature=(
                self._config.temperature
                if temperature is None
                else temperature
            ),
            max_tokens=(
                self._config.max_tokens
                if max_tokens is None
                else max_tokens
            ),
            metadata={} if metadata is None else metadata,
            top_p=self._config.top_p,
            frequency_penalty=self._config.frequency_penalty,
            presence_penalty=self._config.presence_penalty,
        )
