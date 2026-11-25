"""Abstract interface for autonomous agent tools."""

from abc import ABC, abstractmethod

from seven.core.agents.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class Tool(ABC):
    """Define the contract every agent tool must implement."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition and its configuration."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return its result."""
        raise NotImplementedError