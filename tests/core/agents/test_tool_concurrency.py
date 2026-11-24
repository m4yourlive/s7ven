"""Tests for concurrent autonomous agent tool execution."""

import asyncio

import pytest

from seven.core.agents.tools import (
    Tool,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionEngine,
    ToolRegistry,
    ToolResult,
)


class ConcurrentTool(Tool):
    """Track concurrent executions and return their input values."""

    def __init__(self, delay: float = 0.02) -> None:
        self._definition = ToolDefinition(
            name="concurrent_tool",
            description="A tool for concurrency testing.",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                },
                "required": ["value"],
            },
        )
        self.delay = delay
        self.active_count = 0
        self.max_active_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, call: ToolCall) -> ToolResult:
        self.active_count += 1
        self.max_active_count = max(
            self.max_active_count,
            self.active_count,
        )

        try:
            await asyncio.sleep(self.delay)

            return ToolResult(
                call_id=call.call_id,
                tool_name=self.definition.name,
                success=True,
                output=call.arguments["value"],
            )
        finally:
            self.active_count -= 1


class FailingConcurrentTool(ConcurrentTool):
    """A tool that fails during concurrent execution."""

    async def execute(self, call: ToolCall) -> ToolResult:
        if call.arguments["value"] == 2:
            raise RuntimeError("Concurrent execution failure.")

        return await super().execute(call)


def create_engine(
    tool: Tool,
    max_concurrency: int = 5,
) -> ToolExecutionEngine:
    registry = ToolRegistry()
    registry.register(tool)

    return ToolExecutionEngine(
        registry=registry,
        max_concurrency=max_concurrency,
    )


def create_calls(count: int) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(
            tool_name="concurrent_tool",
            arguments={"value": index},
        )
        for index in range(count)
    )


def test_engine_default_max_concurrency():
    registry = ToolRegistry()
    engine = ToolExecutionEngine(registry)

    assert engine.max_concurrency == 5


def test_engine_accepts_custom_max_concurrency():
    registry = ToolRegistry()
    engine = ToolExecutionEngine(
        registry,
        max_concurrency=3,
    )

    assert engine.max_concurrency == 3


def test_engine_rejects_invalid_max_concurrency_type():
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        ToolExecutionEngine(registry, max_concurrency=True)


def test_engine_rejects_zero_max_concurrency():
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        ToolExecutionEngine(registry, max_concurrency=0)


def test_engine_rejects_negative_max_concurrency():
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        ToolExecutionEngine(registry, max_concurrency=-1)


@pytest.mark.asyncio
async def test_execute_many_returns_results_in_input_order():
    tool = ConcurrentTool()
    engine = create_engine(tool, max_concurrency=3)
    calls = create_calls(5)

    results = await engine.execute_many(calls)

    assert len(results) == 5
    assert [result.output for result in results] == list(range(5))


@pytest.mark.asyncio
async def test_execute_many_respects_concurrency_limit():
    tool = ConcurrentTool()
    engine = create_engine(tool, max_concurrency=2)

    results = await engine.execute_many(create_calls(6))

    assert len(results) == 6
    assert tool.max_active_count == 2


@pytest.mark.asyncio
async def test_execute_many_supports_single_call():
    engine = create_engine(ConcurrentTool())
    calls = create_calls(1)

    results = await engine.execute_many(calls)

    assert len(results) == 1
    assert results[0].success is True


@pytest.mark.asyncio
async def test_execute_many_supports_empty_sequence():
    engine = create_engine(ConcurrentTool())

    results = await engine.execute_many(())

    assert results == ()


@pytest.mark.asyncio
async def test_execute_many_preserves_call_ids():
    engine = create_engine(ConcurrentTool())
    calls = create_calls(3)

    results = await engine.execute_many(calls)

    assert [result.call_id for result in results] == [
        call.call_id for call in calls
    ]


@pytest.mark.asyncio
async def test_execute_many_generates_unique_execution_contexts():
    engine = create_engine(ConcurrentTool())
    calls = create_calls(3)

    results = await engine.execute_many(calls)

    execution_ids = [
        result.metadata["execution_context"]["execution_id"]
        for result in results
    ]

    assert len(set(execution_ids)) == 3


@pytest.mark.asyncio
async def test_execute_many_propagates_parent_context():
    engine = create_engine(ConcurrentTool())
    task_id = ToolCall(tool_name="concurrent_tool").call_id

    context = ToolExecutionContext(
        metadata={"source": "batch_test"},
    )

    results = await engine.execute_many(
        create_calls(2),
        context=context,
    )

    for result in results:
        execution_context = result.metadata["execution_context"]

        assert execution_context["task_id"] is None
        assert execution_context["metadata"]["source"] == "batch_test"


@pytest.mark.asyncio
async def test_execute_many_normalizes_individual_failures():
    engine = create_engine(FailingConcurrentTool())
    calls = create_calls(4)

    results = await engine.execute_many(calls)

    assert len(results) == 4
    assert results[0].success is True
    assert results[1].success is True
    assert results[2].success is False
    assert "Concurrent execution failure." in results[2].error
    assert results[3].success is True


@pytest.mark.asyncio
async def test_execute_many_rejects_non_sequence():
    engine = create_engine(ConcurrentTool())

    with pytest.raises(TypeError):
        await engine.execute_many("invalid")


@pytest.mark.asyncio
async def test_execute_many_rejects_invalid_items():
    engine = create_engine(ConcurrentTool())

    with pytest.raises(TypeError):
        await engine.execute_many([object()])


@pytest.mark.asyncio
async def test_execute_many_rejects_invalid_context():
    engine = create_engine(ConcurrentTool())

    with pytest.raises(TypeError):
        await engine.execute_many(
            create_calls(1),
            context=object(),
        )


@pytest.mark.asyncio
async def test_execute_many_propagates_cancellation():
    tool = ConcurrentTool(delay=1)
    engine = create_engine(tool, max_concurrency=2)

    task = asyncio.create_task(
        engine.execute_many(create_calls(4))
    )

    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task