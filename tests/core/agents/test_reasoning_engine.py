
"""Tests for the autonomous reasoning engine."""

from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.models import (
    FunctionCall,
    GenerationResponse,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.agents.models import AgentTask
from seven.core.agents.reasoning import ReasoningStatus
from seven.core.agents.reasoning_config import ReasoningConfig
from seven.core.agents.reasoning_engine import ReasoningEngine
from seven.core.agents.tools.engine import ToolExecutionEngine
from seven.core.agents.tools.interfaces import Tool
from seven.core.agents.tools.models import ToolCall, ToolDefinition, ToolResult
from seven.core.agents.tools.registry import ToolRegistry


class MockTool(Tool):
    """Tool implementation for reasoning engine tests."""

    def __init__(self, requires_confirmation: bool = False) -> None:
        self._definition = ToolDefinition(
            name="echo",
            description="Return the supplied text.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
            requires_confirmation=requires_confirmation,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output=call.arguments.get("text", ""),
        )


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-api-key")


def create_reasoning_engine(
    provider: OpenAIProvider,
    responses: list[GenerationResponse] | None = None,
    tool: Tool | None = None,
    config: ReasoningConfig | None = None,
) -> ReasoningEngine:
    """Create a reasoning engine with a mocked AI response sequence."""
    ai_engine = AIEngine(provider)
    ai_engine.generate = AsyncMock(side_effect=responses or [])  # type: ignore[method-assign]

    registry = ToolRegistry()

    if tool is not None:
        registry.register(tool)

    tool_engine = ToolExecutionEngine(registry)

    return ReasoningEngine(
        ai_engine=ai_engine,
        tool_execution_engine=tool_engine,
        model="gpt-4o-mini",
        config=config,
    )


def create_response(
    content: str = "",
    tool_calls: tuple[FunctionCall, ...] = (),
) -> GenerationResponse:
    """Create a provider response for a reasoning iteration."""
    return GenerationResponse(
        content=content,
        model="gpt-4o-mini",
        provider="openai",
        tool_calls=tool_calls,
    )


@pytest.mark.asyncio
async def test_reasoning_engine_returns_final_response(provider):
    engine = create_reasoning_engine(
        provider,
        responses=[create_response(content="Task completed.")],
    )

    result = await engine.reason(AgentTask(goal="Complete the task."))

    assert result.status is ReasoningStatus.COMPLETED
    assert result.output == "Task completed."
    assert result.iterations == 1
    assert len(result.steps) == 1
    assert result.error is None


@pytest.mark.asyncio
async def test_reasoning_engine_executes_tool_and_continues(provider):
    function_call = FunctionCall(
        call_id="call-1",
        name="echo",
        arguments={"text": "Hello from tool."},
    )

    engine = create_reasoning_engine(
        provider,
        responses=[
            create_response(tool_calls=(function_call,)),
            create_response(content="The tool returned: Hello from tool."),
        ],
        tool=MockTool(),
    )

    result = await engine.reason(AgentTask(goal="Use the echo tool."))

    assert result.status is ReasoningStatus.COMPLETED
    assert result.output == "The tool returned: Hello from tool."
    assert result.iterations == 2
    assert result.metadata["tool_execution_count"] == 1
    assert result.steps[0].tool_call_ids == ("call-1",)


@pytest.mark.asyncio
async def test_reasoning_engine_includes_registered_tool_definitions(provider):
    function_call = FunctionCall(
        call_id="call-1",
        name="echo",
        arguments={"text": "Check definition."},
    )

    engine = create_reasoning_engine(
        provider,
        responses=[
            create_response(tool_calls=(function_call,)),
            create_response(content="Done."),
        ],
        tool=MockTool(),
    )

    result = await engine.reason(AgentTask(goal="Inspect available tools."))

    first_request = engine.ai_engine.generate.await_args_list[0].args[0]

    assert result.status is ReasoningStatus.COMPLETED
    assert len(first_request.tools) == 1
    assert first_request.tools[0].name == "echo"


@pytest.mark.asyncio
async def test_reasoning_engine_blocks_tools_requiring_confirmation(provider):
    function_call = FunctionCall(
        call_id="call-1",
        name="echo",
        arguments={"text": "This must not execute."},
    )

    tool = MockTool(requires_confirmation=True)

    engine = create_reasoning_engine(
        provider,
        responses=[
            create_response(tool_calls=(function_call,)),
            create_response(content="The action was not executed."),
        ],
        tool=tool,
    )

    result = await engine.reason(AgentTask(goal="Run a protected action."))

    assert result.status is ReasoningStatus.COMPLETED

    second_request = engine.ai_engine.generate.await_args_list[1].args[0]
    tool_message = second_request.messages[-1]

    assert tool_message.role.value == "tool"
    assert "requires user confirmation" in tool_message.content
    assert "execution_blocked" in tool_message.content


@pytest.mark.asyncio
async def test_reasoning_engine_stops_at_maximum_iteration_limit(provider):
    function_call = FunctionCall(
        call_id="call-1",
        name="echo",
        arguments={"text": "One iteration."},
    )

    engine = create_reasoning_engine(
        provider,
        responses=[create_response(tool_calls=(function_call,))],
        tool=MockTool(),
        config=ReasoningConfig(max_iterations=1),
    )

    result = await engine.reason(AgentTask(goal="Continue until complete."))

    assert result.status is ReasoningStatus.LIMIT_REACHED
    assert result.iterations == 1
    assert "maximum iteration limit" in result.output


@pytest.mark.asyncio
async def test_reasoning_engine_stops_at_total_tool_call_limit(provider):
    function_calls = (
        FunctionCall(
            call_id="call-1",
            name="echo",
            arguments={"text": "First call."},
        ),
        FunctionCall(
            call_id="call-2",
            name="echo",
            arguments={"text": "Second call."},
        ),
    )

    engine = create_reasoning_engine(
        provider,
        responses=[create_response(tool_calls=function_calls)],
        tool=MockTool(),
        config=ReasoningConfig(max_tool_calls=1),
    )

    result = await engine.reason(AgentTask(goal="Call tools."))

    assert result.status is ReasoningStatus.LIMIT_REACHED
    assert "total tool call limit" in result.output


@pytest.mark.asyncio
async def test_reasoning_engine_returns_failed_result_on_provider_error(provider):
    engine = create_reasoning_engine(provider)
    engine.ai_engine.generate = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("Provider connection failed.")
    )

    result = await engine.reason(AgentTask(goal="Generate a response."))

    assert result.status is ReasoningStatus.FAILED
    assert result.error is not None
    assert "Provider connection failed." in result.error
