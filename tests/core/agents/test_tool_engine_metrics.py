"""Integration tests for execution engine lifecycle metrics."""

import asyncio

import pytest

from seven.core.agents.tools import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionEngine,
    ToolExecutionStatus,
    ToolRegistry,
    ToolResult,
)


class MetricsTool(Tool):
    """Tool used to test execution metrics."""

    def __init__(
        self,
        timeout: float = 1.0,
        should_fail: bool = False,
        delay: float = 0.01,
    ) -> None:
        self._definition = ToolDefinition(
            name="metrics_tool",
            description="A tool for metrics testing.",
            timeout_seconds=timeout,
        )
        self.should_fail = should_fail
        self.delay = delay

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, call: ToolCall) -> ToolResult:
        await asyncio.sleep(self.delay)

        if self.should_fail:
            raise RuntimeError("Metrics test failure.")

        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output="Completed.",
        )


def create_engine(tool: Tool) -> ToolExecutionEngine:
    registry = ToolRegistry()
    registry.register(tool)

    return ToolExecutionEngine(registry)


@pytest.mark.asyncio
async def test_engine_tracks_successful_execution():
    engine = create_engine(MetricsTool())
    call = ToolCall(tool_name="metrics_tool")

    result = await engine.execute(call)

    assert result.success is True
    assert engine.metrics.total_executions == 1
    assert engine.metrics.succeeded == 1
    assert engine.metrics.active == 0


@pytest.mark.asyncio
async def test_engine_tracks_failed_execution():
    engine = create_engine(MetricsTool(should_fail=True))

    result = await engine.execute(
        ToolCall(tool_name="metrics_tool")
    )

    assert result.success is False
    assert engine.metrics.failed == 1


@pytest.mark.asyncio
async def test_engine_tracks_timeout():
    engine = create_engine(
        MetricsTool(timeout=0.001, delay=0.1)
    )

    result = await engine.execute(
        ToolCall(tool_name="metrics_tool")
    )

    assert result.success is False
    assert engine.metrics.timed_out == 1


@pytest.mark.asyncio
async def test_engine_tracks_unknown_tool_failure():
    engine = ToolExecutionEngine(ToolRegistry())

    result = await engine.execute(
        ToolCall(tool_name="unknown_tool")
    )

    assert result.success is False
    assert engine.metrics.failed == 1


@pytest.mark.asyncio
async def test_engine_tracks_validation_failure():
    class StrictTool(MetricsTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="metrics_tool",
                description="A strict test tool.",
                parameters={
                    "type": "object",
                    "required": ["query"],
                },
            )

    engine = create_engine(StrictTool())

    result = await engine.execute(
        ToolCall(tool_name="metrics_tool")
    )

    assert result.success is False
    assert engine.metrics.failed == 1


@pytest.mark.asyncio
async def test_engine_exposes_execution_record():
    engine = create_engine(MetricsTool())
    call = ToolCall(tool_name="metrics_tool")

    result = await engine.execute(call)
    execution_id = result.metadata["execution_context"]["execution_id"]

    from uuid import UUID

    record = engine.tracker.get_record(UUID(execution_id))

    assert record is not None
    assert record.status == ToolExecutionStatus.SUCCEEDED
    assert record.call_id == call.call_id
    assert record.completed_at is not None


@pytest.mark.asyncio
async def test_engine_tracks_concurrent_batch_metrics():
    engine = create_engine(MetricsTool())
    calls = tuple(
        ToolCall(tool_name="metrics_tool")
        for _ in range(4)
    )

    results = await engine.execute_many(calls)

    assert len(results) == 4
    assert engine.metrics.total_executions == 4
    assert engine.metrics.succeeded == 4
    assert engine.metrics.active == 0


@pytest.mark.asyncio
async def test_engine_tracks_cancelled_execution():
    engine = create_engine(
        MetricsTool(timeout=5, delay=1)
    )

    task = asyncio.create_task(
        engine.execute(ToolCall(tool_name="metrics_tool"))
    )

    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.metrics.cancelled == 1
    assert engine.metrics.active == 0


@pytest.mark.asyncio
async def test_engine_rejects_invalid_tracker():
    with pytest.raises(TypeError):
        ToolExecutionEngine(
            ToolRegistry(),
            tracker=object(),
        )