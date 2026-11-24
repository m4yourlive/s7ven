
"""Data models for the S7VEN AI engine."""

from dataclasses import dataclass, field
from enum import StrEnum

class MessageRole(StrEnum):
    """Supported conversation message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Describe the generation features supported by a provider."""

    supports_streaming: bool = True
    supports_system_messages: bool = True
    supports_temperature: bool = True
    supports_max_tokens: bool = True
    supports_top_p: bool = True
    supports_frequency_penalty: bool = True
    supports_presence_penalty: bool = True

    def __post_init__(self) -> None:
        """Validate provider capability flags."""
        for field_name in (
            "supports_streaming",
            "supports_system_messages",
            "supports_temperature",
            "supports_max_tokens",
            "supports_top_p",
            "supports_frequency_penalty",
            "supports_presence_penalty",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(
                    f"{field_name} must be a boolean."
                )

@dataclass(frozen=True, slots=True)
class AIMessage:
    """A single message in an AI conversation."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole.")

        if not isinstance(self.content, str):
            raise TypeError("content must be a string.")

        if not self.content.strip():
            raise ValueError("Message content cannot be empty.")

@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Input parameters for an AI generation request."""

    messages: tuple[AIMessage, ...]
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple.")

        if not self.messages:
            raise ValueError("At least one message is required.")

        if not all(
            isinstance(message, AIMessage)
            for message in self.messages
        ):
            raise TypeError("All messages must be AIMessage instances.")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model name cannot be empty.")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero.")

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1.")

        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be between -2.0 and 2.0."
            )

        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be between -2.0 and 2.0."
            )

@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """Output returned by an AI provider."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Response content must be a string.")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Response model cannot be empty.")

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("Provider name cannot be empty.")