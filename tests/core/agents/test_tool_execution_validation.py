"""Integration tests for execution context and argument validation."""

from uuid import uuid4

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


class SearchTool(Tool):
    """Mock search tool with a strict argument schema."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search",
            description="Search for a query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 2,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name="search",
            success=True,
            output=f"Results for {call.arguments['query']}",
            metadata={"source": "search_tool"},
        )


@pytest.fixture
def engine():
    registry = ToolRegistry()
    registry.register(SearchTool())
    return ToolExecutionEngine(registry)


@pytest.mark.asyncio
async def test_engine_executes_valid_arguments(engine):
    call = ToolCall(
        tool_name="search",
        arguments={"query": "Bitcoin"},
    )

    result = await engine.execute(call)

    assert result.success is True
    assert result.output == "Results for Bitcoin"


@pytest.mark.asyncio
async def test_engine_rejects_missing_required_argument(engine):
    call = ToolCall(
        tool_name="search",
        arguments={},
    )

    result = await engine.execute(call)

    assert result.success is False
    assert result.metadata["validation_error"] is True
    assert "query is required" in result.error


@pytest.mark.asyncio
async def test_engine_rejects_invalid_argument_type(engine):
    call = ToolCall(
        tool_name="search",
        arguments={"query": 123},
    )

    result = await engine.execute(call)

    assert result.success is False
    assert result.metadata["validation_error"] is True


@pytest.mark.asyncio
async def test_engine_rejects_additional_arguments(engine):
    call = ToolCall(
        tool_name="search",
        arguments={
            "query": "Bitcoin",
            "unexpected": True,
        },
    )

    result = await engine.execute(call)

    assert result.success is False
    assert result.metadata["validation_error"] is True


@pytest.mark.asyncio
async def test_engine_attaches_execution_context(engine):
    task_id = uuid4()
    context = ToolExecutionContext(
        task_id=task_id,
        metadata={"source": "autonomous_agent"},
    )
    call = ToolCall(
        tool_name="search",
        arguments={"query": "Bitcoin"},
    )

    result = await engine.execute(call, context)

    execution_context = result.metadata["execution_context"]

    assert execution_context["task_id"] == str(task_id)
    assert execution_context["execution_id"] == str(context.execution_id)
    assert execution_context["metadata"]["source"] == "autonomous_agent"


@pytest.mark.asyncio
async def test_engine_attaches_context_to_validation_failure(engine):
    context = ToolExecutionContext(
        metadata={"source": "test"},
    )
    call = ToolCall(
        tool_name="search",
        arguments={},
    )

    result = await engine.execute(call, context)

    assert result.success is False
    assert result.metadata["validation_error"] is True
    assert result.metadata["execution_context"]["execution_id"] == str(
        context.execution_id
    )


@pytest.mark.asyncio
async def test_engine_preserves_tool_metadata(engine):
    call = ToolCall(
        tool_name="search",
        arguments={"query": "Bitcoin"},
    )

    result = await engine.execute(call)

    assert result.metadata["source"] == "search_tool"
    assert "execution_context" in result.metadata


@pytest.mark.asyncio
async def test_engine_rejects_invalid_context(engine):
    call = ToolCall(
        tool_name="search",
        arguments={"query": "Bitcoin"},
    )

    with pytest.raises(TypeError):
        await engine.execute(call, context=object())