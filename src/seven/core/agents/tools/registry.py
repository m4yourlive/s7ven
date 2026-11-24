"""Registry for managing autonomous agent tools."""

from seven.core.agents.tools.exceptions import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from seven.core.agents.tools.interfaces import Tool
from seven.core.agents.tools.models import ToolDefinition


class ToolRegistry:
    """Manage the tools available to the autonomous agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @property
    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def register(self, tool: Tool) -> None:
        """Register a tool using its definition name."""
        if not isinstance(tool, Tool):
            raise TypeError("tool must implement the Tool interface.")

        definition = tool.definition

        if not isinstance(definition, ToolDefinition):
            raise ToolRegistrationError(
                "Tool definition must be a ToolDefinition instance."
            )

        name = definition.name

        if name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool '{name}' is already registered."
            )

        self._tools[name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a registered tool by name."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name cannot be empty.")

        if name not in self._tools:
            return False

        del self._tools[name]
        return True

    def get(self, name: str) -> Tool:
        """Retrieve a registered tool by name."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name cannot be empty.")

        tool = self._tools.get(name)

        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered."
            )

        return tool

    def get_definition(self, name: str) -> ToolDefinition:
        """Retrieve a registered tool definition by name."""
        return self.get(name).definition

    def contains(self, name: str) -> bool:
        """Check whether a tool name is registered."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name cannot be empty.")

        return name in self._tools

    def list_tools(self) -> tuple[Tool, ...]:
        """Return all registered tools."""
        return tuple(self._tools.values())

    def list_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the definitions of all registered tools."""
        return tuple(
            tool.definition
            for tool in self._tools.values()
        )

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()