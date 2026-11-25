
"""Data models for the S7VEN AI engine."""

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class MessageRole(StrEnum):
    """Supported conversation message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolChoice(StrEnum):
    """Control how an AI provider selects available functions."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


def _freeze_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and freeze a mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")

    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings.")

    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    """Describe a function that an AI provider can call."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Function name cannot be empty.")

        if not isinstance(self.description, str):
            raise TypeError("Function description must be a string.")

        if not self.description.strip():
            raise ValueError("Function description cannot be empty.")

        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """Represent a function call requested by an AI provider."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("Function call ID cannot be empty.")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Function name cannot be empty.")

        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(self.arguments, "arguments"),
        )


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
    supports_function_calling: bool = False

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
            "supports_function_calling",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class AIMessage:
    """A single message in an AI conversation."""

    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[FunctionCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole.")

        if not isinstance(self.content, str):
            raise TypeError("content must be a string.")

        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple.")

        if not all(
            isinstance(call, FunctionCall)
            for call in self.tool_calls
        ):
            raise TypeError(
                "All tool_calls must be FunctionCall instances."
            )

        if self.role == MessageRole.TOOL:
            if (
                not isinstance(self.tool_call_id, str)
                or not self.tool_call_id.strip()
            ):
                raise ValueError(
                    "Tool messages require a valid tool_call_id."
                )

            if not self.content.strip():
                raise ValueError("Tool message content cannot be empty.")

            if self.tool_calls:
                raise ValueError(
                    "Tool messages cannot contain tool_calls."
                )

        else:
            if self.tool_call_id is not None:
                raise ValueError(
                    "tool_call_id is only valid for tool messages."
                )

            if self.tool_calls and self.role != MessageRole.ASSISTANT:
                raise ValueError(
                    "Only assistant messages can contain tool_calls."
                )

            if not self.content.strip() and not self.tool_calls:
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
    tools: tuple[FunctionDefinition, ...] = ()
    tool_choice: ToolChoice = ToolChoice.AUTO

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

        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple.")

        if not all(
            isinstance(tool, FunctionDefinition)
            for tool in self.tools
        ):
            raise TypeError(
                "All tools must be FunctionDefinition instances."
            )

        tool_names = [tool.name for tool in self.tools]

        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Function names must be unique.")

        if not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool_choice must be a ToolChoice.")

        if (
            self.tool_choice == ToolChoice.REQUIRED
            and not self.tools
        ):
            raise ValueError(
                "Required tool choice needs at least one function."
            )


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """Output returned by an AI provider."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: tuple[FunctionCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Response content must be a string.")

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Response model cannot be empty.")

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("Provider name cannot be empty.")

        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple.")

        if not all(
            isinstance(call, FunctionCall)
            for call in self.tool_calls
        ):
            raise TypeError(
                "All tool_calls must be FunctionCall instances."
            )

        if not self.content.strip() and not self.tool_calls:
            raise ValueError(
                "Response must contain text or at least one tool call."
            )
