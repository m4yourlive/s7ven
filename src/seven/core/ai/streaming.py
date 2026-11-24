
"""Streaming event models for the S7VEN AI system."""

from dataclasses import dataclass

from seven.core.ai.models import FunctionCall


@dataclass(frozen=True, slots=True)
class AIStreamEvent:
    """Represent a normalized event emitted by an AI provider stream."""

    content: str = ""
    tool_calls: tuple[FunctionCall, ...] = ()
    is_final: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Stream content must be a string.")

        if not isinstance(self.tool_calls, tuple):
            raise TypeError("Stream tool calls must be a tuple.")

        if not all(
            isinstance(call, FunctionCall)
            for call in self.tool_calls
        ):
            raise TypeError(
                "Stream tool calls must contain FunctionCall instances."
            )

        if not isinstance(self.is_final, bool):
            raise TypeError("Stream final flag must be a boolean.")

        if self.content and self.tool_calls:
            raise ValueError(
                "A stream event cannot contain content and tool calls together."
            )

        if self.is_final and self.content:
            raise ValueError(
                "A final stream event cannot contain text content."
            )
