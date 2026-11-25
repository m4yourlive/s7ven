"""Tests for autonomous agent tool execution events."""

import asyncio

import pytest

from seven.core.agents.tools import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionCompleted,
    ToolExecutionEngine,
    ToolExecutionFailed,
    ToolExecutionStarted,
    ToolExecutionTimedOut,
    ToolRegistry,
    ToolResult,
)
from seven.core.events import EventBus


class SuccessfulTool(Tool):
    """Tool that completes successfully."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="successful_tool",
            description="A successful test tool.",
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output="Completed.",
        )


class FailingTool(Tool):
    """Tool that raises an exception."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failing_tool",
            description="A failing test tool.",
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        raise RuntimeError("Test execution failure.")


class SlowTool(Tool):
    """Tool that exceeds its timeout."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="slow_tool",
            description="A slow test tool.",
            timeout_seconds=0.01,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
        )


def create_engine(tool: Tool, event_bus: EventBus):
    registry = ToolRegistry()
    registry.register(tool)

    return ToolExecutionEngine(
        registry=registry,
        event_bus=event_bus,
    )


@pytest.mark.asyncio
async def test_engine_publishes_started_event():
    event_bus = EventBus()
    events = []

    event_bus.subscribe(
        ToolExecutionStarted,
        events.append,
    )

    engine = create_engine(SuccessfulTool(), event_bus)
    call = ToolCall(tool_name="successful_tool")

    await engine.execute(call)

    assert len(events) == 1
    assert events[0].call == call


@pytest.mark.asyncio
async def test_engine_publishes_completed_event():
    event_bus = EventBus()
    events = []

    event_bus.subscribe(
        ToolExecutionCompleted,
        events.append,
    )

    engine = create_engine(SuccessfulTool(), event_bus)

    result = await engine.execute(
        ToolCall(tool_name="successful_tool")
    )

    assert result.success is True
    assert len(events) == 1
    assert events[0].result == result


@pytest.mark.asyncio
async def test_engine_publishes_failed_event():
    event_bus = EventBus()
    events = []

    event_bus.subscribe(
        ToolExecutionFailed,
        events.append,
    )

    engine = create_engine(FailingTool(), event_bus)

    result = await engine.execute(
        ToolCall(tool_name="failing_tool")
    )

    assert result.success is False
    assert len(events) == 1
    assert events[0].result == result


@pytest.mark.asyncio
async def test_engine_publishes_timeout_event():
    event_bus = EventBus()
    events = []

    event_bus.subscribe(
        ToolExecutionTimedOut,
        events.append,
    )

    engine = create_engine(SlowTool(), event_bus)

    result = await engine.execute(
        ToolCall(tool_name="slow_tool")
    )

    assert result.success is False
    assert result.metadata["timeout"] is True
    assert len(events) == 1
    assert events[0].result == result


@pytest.mark.asyncio
async def test_engine_publishes_started_before_completed():
    event_bus = EventBus()
    event_types = []

    event_bus.subscribe(
        ToolExecutionStarted,
        lambda event: event_types.append(type(event)),
    )
    event_bus.subscribe(
        ToolExecutionCompleted,
        lambda event: event_types.append(type(event)),
    )

    engine = create_engine(SuccessfulTool(), event_bus)

    await engine.execute(
        ToolCall(tool_name="successful_tool")
    )

    assert event_types == [
        ToolExecutionStarted,
        ToolExecutionCompleted,
    ]


@pytest.mark.asyncio
async def test_execution_events_share_execution_id():
    event_bus = EventBus()
    events = []

    event_bus.subscribe(
        ToolExecutionStarted,
        events.append,
    )
    event_bus.subscribe(
        ToolExecutionCompleted,
        events.append,
    )

    engine = create_engine(SuccessfulTool(), event_bus)

    await engine.execute(
        ToolCall(tool_name="successful_tool")
    )

    started_event, completed_event = events

    assert (
        started_event.context.execution_id
        == completed_event.execution_id
    )


@pytest.mark.asyncio
async def test_engine_publishes_failure_for_unknown_tool():
    event_bus = EventBus()
    failed_events = []

    event_bus.subscribe(
        ToolExecutionFailed,
        failed_events.append,
    )

    engine = ToolExecutionEngine(
        registry=ToolRegistry(),
        event_bus=event_bus,
    )

    result = await engine.execute(
        ToolCall(tool_name="unknown_tool")
    )

    assert result.success is False
    assert len(failed_events) == 1


def test_engine_accepts_optional_event_bus():
    engine = ToolExecutionEngine(
        registry=ToolRegistry(),
    )

    assert engine.event_bus is None


def test_engine_rejects_invalid_event_bus():
    with pytest.raises(TypeError):
        ToolExecutionEngine(
            registry=ToolRegistry(),
            event_bus=object(),
        )


@pytest.mark.asyncio
async def test_subscriber_failure_does_not_break_tool_execution():
    event_bus = EventBus()

    def failing_subscriber(event):
        raise RuntimeError("Subscriber failure.")

    event_bus.subscribe(
        ToolExecutionStarted,
        failing_subscriber,
    )

    engine = create_engine(SuccessfulTool(), event_bus)

    result = await engine.execute(
        ToolCall(tool_name="successful_tool")
    )

    assert result.success is True