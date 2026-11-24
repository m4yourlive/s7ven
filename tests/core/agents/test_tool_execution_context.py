"""Tests for tool execution context."""

from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import pytest

from seven.core.agents.tools import ToolExecutionContext


def test_context_generates_unique_execution_id():
    first = ToolExecutionContext()
    second = ToolExecutionContext()

    assert isinstance(first.execution_id, UUID)
    assert first.execution_id != second.execution_id


def test_context_accepts_task_id():
    task_id = uuid4()
    context = ToolExecutionContext(task_id=task_id)

    assert context.task_id == task_id


def test_context_accepts_metadata():
    context = ToolExecutionContext(
        metadata={"source": "agent"},
    )

    assert context.metadata["source"] == "agent"


def test_context_metadata_is_immutable():
    context = ToolExecutionContext(
        metadata={"source": "agent"},
    )

    with pytest.raises(TypeError):
        context.metadata["source"] = "external"


def test_context_is_immutable():
    context = ToolExecutionContext()

    with pytest.raises(FrozenInstanceError):
        context.execution_id = uuid4()


def test_context_rejects_invalid_task_id():
    with pytest.raises(TypeError):
        ToolExecutionContext(task_id="invalid")


def test_context_rejects_invalid_metadata():
    with pytest.raises(TypeError):
        ToolExecutionContext(metadata=[])


def test_context_rejects_non_string_metadata_keys():
    with pytest.raises(TypeError):
        ToolExecutionContext(metadata={1: "invalid"})