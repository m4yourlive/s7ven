"""Integration tests for the autonomous agent tool system."""

from seven.core.agents.tools import (
    Tool,
    ToolAlreadyRegisteredError,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)


class EchoTool(Tool):
    """Return the input message as the tool output."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Return the supplied message.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        message = call.arguments.get("message", "")

        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output=message,
            metadata={"tool": self.definition.name},
        )


def test_tool_system_public_exports():
    from seven.core.agents.tools import (
        Tool,
        ToolCall,
        ToolDefinition,
        ToolRegistry,
        ToolResult,
    )

    assert Tool is not None
    assert ToolCall is not None
    assert ToolDefinition is not None
    assert ToolRegistry is not None
    assert ToolResult is not None


def test_tool_can_be_registered_and_retrieved():
    registry = ToolRegistry()
    tool = EchoTool()

    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.get_definition("echo") == tool.definition


def test_registered_tool_can_be_executed():
    import asyncio

    registry = ToolRegistry()
    registry.register(EchoTool())

    tool = registry.get("echo")
    call = ToolCall(
        tool_name="echo",
        arguments={"message": "Hello from S7VEN."},
    )

    result = asyncio.run(tool.execute(call))

    assert result.success is True
    assert result.output == "Hello from S7VEN."
    assert result.call_id == call.call_id
    assert result.tool_name == "echo"


def test_tool_execution_result_preserves_metadata():
    import asyncio

    registry = ToolRegistry()
    registry.register(EchoTool())

    call = ToolCall(
        tool_name="echo",
        arguments={"message": "Integration test."},
    )

    result = asyncio.run(registry.get("echo").execute(call))

    assert result.metadata["tool"] == "echo"


def test_registry_can_replace_tool_after_unregistering():
    registry = ToolRegistry()
    first_tool = EchoTool()

    registry.register(first_tool)
    assert registry.unregister("echo") is True

    replacement = EchoTool()
    registry.register(replacement)

    assert registry.get("echo") is replacement
    assert registry.count == 1


def test_registry_rejects_duplicate_tool_registration():
    registry = ToolRegistry()
    registry.register(EchoTool())

    try:
        registry.register(EchoTool())
    except ToolAlreadyRegisteredError:
        pass
    else:
        raise AssertionError("Duplicate tool registration was not rejected.")