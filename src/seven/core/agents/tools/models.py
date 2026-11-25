
"""Domain models for the S7VEN agent tool system."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4


def _freeze_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")

    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings.")

    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition:
    """Describe a tool available to the autonomous agent."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name cannot be empty.")

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Tool description cannot be empty.")

        if not isinstance(self.requires_confirmation, bool):
            raise TypeError("requires_confirmation must be a boolean.")

        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
        ):
            raise TypeError("timeout_seconds must be a number.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """Represent a request to execute a registered tool."""

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, UUID):
            raise TypeError("call_id must be a UUID.")

        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(self.arguments, "arguments"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult:
    """Represent the outcome of a tool execution."""

    call_id: UUID
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, UUID):
            raise TypeError("call_id must be a UUID.")

        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean.")

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")

        if self.success and self.error:
            raise ValueError("A successful tool result cannot contain an error.")

        if not self.success and not self.error:
            raise ValueError("A failed tool result must contain an error.")

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )
