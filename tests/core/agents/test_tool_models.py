
"""Tests for autonomous agent tool domain models."""

from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import pytest

from seven.core.agents.tools import (
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def test_tool_definition_creation():
    tool = ToolDefinition(
        name="web_search",
        description="Search the web for information.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
        },
    )

    assert tool.name == "web_search"
    assert tool.description == "Search the web for information."
    assert tool.parameters["type"] == "object"
    assert tool.requires_confirmation is False
    assert tool.timeout_seconds == 30.0


def test_tool_definition_rejects_empty_name():
    with pytest.raises(ValueError):
        ToolDefinition(
            name=" ",
            description="A valid description.",
        )


def test_tool_definition_rejects_empty_description():
    with pytest.raises(ValueError):
        ToolDefinition(
            name="web_search",
            description=" ",
        )


def test_tool_definition_rejects_invalid_confirmation_flag():
    with pytest.raises(TypeError):
        ToolDefinition(
            name="web_search",
            description="Search the web.",
            requires_confirmation="yes",
        )


def test_tool_definition_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        ToolDefinition(
            name="web_search",
            description="Search the web.",
            timeout_seconds=0,
        )


def test_tool_definition_rejects_non_mapping_parameters():
    with pytest.raises(TypeError):
        ToolDefinition(
            name="web_search",
            description="Search the web.",
            parameters=[],
        )


def test_tool_definition_parameters_are_immutable():
    tool = ToolDefinition(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object"},
    )

    with pytest.raises(TypeError):
        tool.parameters["type"] = "string"


def test_tool_call_creation():
    call = ToolCall(
        tool_name="web_search",
        arguments={"query": "Bitcoin"},
    )

    assert isinstance(call.call_id, UUID)
    assert call.tool_name == "web_search"
    assert call.arguments["query"] == "Bitcoin"


def test_tool_calls_have_unique_ids():
    first = ToolCall(tool_name="web_search")
    second = ToolCall(tool_name="web_search")

    assert first.call_id != second.call_id


def test_tool_call_rejects_empty_name():
    with pytest.raises(ValueError):
        ToolCall(tool_name=" ")


def test_tool_call_rejects_invalid_arguments():
    with pytest.raises(TypeError):
        ToolCall(
            tool_name="web_search",
            arguments=[],
        )


def test_tool_call_arguments_are_immutable():
    call = ToolCall(
        tool_name="web_search",
        arguments={"query": "Bitcoin"},
    )

    with pytest.raises(TypeError):
        call.arguments["query"] = "Ethereum"


def test_tool_result_creation():
    call_id = uuid4()

    result = ToolResult(
        call_id=call_id,
        tool_name="web_search",
        success=True,
        output={"results": ["Result A", "Result B"]},
    )

    assert result.call_id == call_id
    assert result.tool_name == "web_search"
    assert result.success is True
    assert len(result.output["results"]) == 2
    assert result.error is None


def test_failed_tool_result_requires_error():
    with pytest.raises(ValueError):
        ToolResult(
            call_id=uuid4(),
            tool_name="web_search",
            success=False,
        )


def test_successful_tool_result_rejects_error():
    with pytest.raises(ValueError):
        ToolResult(
            call_id=uuid4(),
            tool_name="web_search",
            success=True,
            error="Unexpected failure",
        )


def test_tool_result_rejects_invalid_success_flag():
    with pytest.raises(TypeError):
        ToolResult(
            call_id=uuid4(),
            tool_name="web_search",
            success="true",
        )


def test_tool_result_metadata_is_immutable():
    result = ToolResult(
        call_id=uuid4(),
        tool_name="web_search",
        success=True,
        output="Completed",
        metadata={"source": "test"},
    )

    with pytest.raises(TypeError):
        result.metadata["source"] = "changed"


def test_tool_models_are_immutable():
    definition = ToolDefinition(
        name="web_search",
        description="Search the web.",
    )

    call = ToolCall(tool_name="web_search")

    with pytest.raises(FrozenInstanceError):
        definition.name = "other_tool"

    with pytest.raises(FrozenInstanceError):
        call.tool_name = "other_tool"
