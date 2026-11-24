"""Tests for the autonomous agent tool execution engine."""

import asyncio

import pytest

from seven.core.agents.tools import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionEngine,
    ToolRegistry,
    ToolResult,
)


class MockTool(Tool):
    """Tool implementation for execution engine tests."""

    def __init__(
        self,
        name: str = "mock_tool",
        timeout: float = 1.0,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description="A mock tool for execution testing.",
            timeout_seconds=timeout,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output="Execution completed.",
        )


class FailingTool(MockTool):
    """Tool that raises an exception during execution."""

    async def execute(self, call: ToolCall) -> ToolResult:
        raise RuntimeError("Mock execution failure.")


class SlowTool(MockTool):
    """Tool that exceeds its configured timeout."""

    async def execute(self, call: ToolCall) -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output="Completed.",
        )


class InvalidResultTool(MockTool):
    """Tool that returns an invalid result type."""

    async def execute(self, call: ToolCall):
        return "Invalid result."


class MismatchedCallTool(MockTool):
    """Tool that returns a result with a different call ID."""

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=ToolCall(tool_name=self.definition.name).call_id,
            tool_name=self.definition.name,
            success=True,
            output="Completed.",
        )


def create_engine(tool: Tool | None = None) -> ToolExecutionEngine:
    registry = ToolRegistry()

    if tool is not None:
        registry.register(tool)

    return ToolExecutionEngine(registry)


def test_engine_requires_tool_registry():
    with pytest.raises(TypeError):
        ToolExecutionEngine(object())


def test_engine_exposes_registry():
    registry = ToolRegistry()
    engine = ToolExecutionEngine(registry)

    assert engine.registry is registry


@pytest.mark.asyncio
async def test_execute_returns_successful_result():
    engine = create_engine(MockTool())
    call = ToolCall(tool_name="mock_tool")

    result = await engine.execute(call)

    assert result.success is True
    assert result.output == "Execution completed."
    assert result.call_id == call.call_id


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_failure():
    engine = create_engine()
    call = ToolCall(tool_name="unknown_tool")

    result = await engine.execute(call)

    assert result.success is False
    assert result.call_id == call.call_id
    assert result.tool_name == "unknown_tool"
    assert "not registered" in result.error


@pytest.mark.asyncio
async def test_execute_converts_tool_exception_into_failure():
    engine = create_engine(FailingTool())
    call = ToolCall(tool_name="mock_tool")

    result = await engine.execute(call)

    assert result.success is False
    assert "Mock execution failure." in result.error


@pytest.mark.asyncio
async def test_execute_returns_failure_when_tool_times_out():
    engine = create_engine(SlowTool(timeout=0.01))
    call = ToolCall(tool_name="mock_tool")

    result = await engine.execute(call)

    assert result.success is False
    assert result.metadata["timeout"] is True
    assert "exceeded its execution timeout" in result.error


@pytest.mark.asyncio
async def test_execute_rejects_invalid_tool_result():
    engine = create_engine(InvalidResultTool())
    call = ToolCall(tool_name="mock_tool")

    result = await engine.execute(call)

    assert result.success is False
    assert "ToolResult instance" in result.error


@pytest.mark.asyncio
async def test_execute_rejects_mismatched_call_id():
    engine = create_engine(MismatchedCallTool())
    call = ToolCall(tool_name="mock_tool")

    result = await engine.execute(call)

    assert result.success is False
    assert "call ID does not match" in result.error


@pytest.mark.asyncio
async def test_execute_rejects_non_tool_call():
    engine = create_engine(MockTool())

    with pytest.raises(TypeError):
        await engine.execute(object())


@pytest.mark.asyncio
async def test_execute_propagates_cancellation():
    engine = create_engine(SlowTool(timeout=5))
    call = ToolCall(tool_name="mock_tool")

    task = asyncio.create_task(engine.execute(call))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task