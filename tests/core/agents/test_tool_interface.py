"""Tests for the autonomous agent tool interface."""

import pytest

from seven.core.agents.tools import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class MockTool(Tool):
    """Concrete tool implementation for interface testing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mock_tool",
            description="A mock tool for testing.",
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output="Mock execution completed.",
        )


def test_tool_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Tool()


def test_tool_implementation_exposes_definition():
    tool = MockTool()

    assert isinstance(tool.definition, ToolDefinition)
    assert tool.definition.name == "mock_tool"


def test_tool_implementation_has_correct_description():
    tool = MockTool()

    assert tool.definition.description == "A mock tool for testing."


@pytest.mark.asyncio
async def test_tool_execution_returns_tool_result():
    tool = MockTool()
    call = ToolCall(
        tool_name="mock_tool",
        arguments={},
    )

    result = await tool.execute(call)

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.output == "Mock execution completed."


@pytest.mark.asyncio
async def test_tool_execution_preserves_call_id():
    tool = MockTool()
    call = ToolCall(tool_name="mock_tool")

    result = await tool.execute(call)

    assert result.call_id == call.call_id


@pytest.mark.asyncio
async def test_tool_execution_uses_definition_name():
    tool = MockTool()
    call = ToolCall(tool_name="mock_tool")

    result = await tool.execute(call)

    assert result.tool_name == tool.definition.name


def test_incomplete_tool_implementation_is_rejected():
    class IncompleteTool(Tool):
        pass

    with pytest.raises(TypeError):
        IncompleteTool()


def test_tool_is_exported_from_package():
    from seven.core.agents.tools.interfaces import Tool as InterfaceTool

    assert Tool is InterfaceTool