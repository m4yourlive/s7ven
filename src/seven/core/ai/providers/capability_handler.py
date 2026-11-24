
"""Shared capability validation and generation option handling."""

from collections.abc import Sequence
from typing import Any

from seven.core.ai.exceptions import AIProviderCapabilityError
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    ProviderCapabilities,
)


class CapabilityHandler:
    """Validate provider capabilities and filter generation options."""

    @staticmethod
    def validate_system_messages(
        messages: Sequence[AIMessage],
        capabilities: ProviderCapabilities,
        provider_name: str,
    ) -> None:
        """Reject system messages when the provider does not support them."""
        if not isinstance(capabilities, ProviderCapabilities):
            raise TypeError(
                "capabilities must be a ProviderCapabilities instance."
            )

        if not capabilities.supports_system_messages and any(
            message.role.value == "system" for message in messages
        ):
            raise AIProviderCapabilityError(
                f"{provider_name} does not support system messages."
            )

    @staticmethod
    def validate_streaming(
        capabilities: ProviderCapabilities,
        provider_name: str,
    ) -> None:
        """Reject streaming when the provider does not support it."""
        if not isinstance(capabilities, ProviderCapabilities):
            raise TypeError(
                "capabilities must be a ProviderCapabilities instance."
            )

        if not capabilities.supports_streaming:
            raise AIProviderCapabilityError(
                f"{provider_name} does not support streaming."
            )

    @staticmethod
    def build_generation_options(
        request: GenerationRequest,
        capabilities: ProviderCapabilities,
        *,
        max_tokens_key: str = "max_tokens",
        include_unset_max_tokens: bool = True,
    ) -> dict[str, Any]:
        """Build generation options supported by the provider."""
        if not isinstance(request, GenerationRequest):
            raise TypeError(
                "request must be a GenerationRequest instance."
            )

        if not isinstance(capabilities, ProviderCapabilities):
            raise TypeError(
                "capabilities must be a ProviderCapabilities instance."
            )

        if (
            not isinstance(max_tokens_key, str)
            or not max_tokens_key.strip()
        ):
            raise ValueError("max_tokens_key cannot be empty.")

        options: dict[str, Any] = {}

        if capabilities.supports_temperature:
            options["temperature"] = request.temperature

        if capabilities.supports_max_tokens:
            if (
                request.max_tokens is not None
                or include_unset_max_tokens
            ):
                options[max_tokens_key] = request.max_tokens

        if capabilities.supports_top_p:
            options["top_p"] = request.top_p

        if capabilities.supports_frequency_penalty:
            options["frequency_penalty"] = request.frequency_penalty

        if capabilities.supports_presence_penalty:
            options["presence_penalty"] = request.presence_penalty

        return options
