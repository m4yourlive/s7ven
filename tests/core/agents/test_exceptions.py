"""Tests for autonomous agent exception hierarchy."""

import pytest

from seven.core.agents.exceptions import (
    AgentCancelledError,
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentPlanningError,
    AgentStateError,
    AgentTaskError,
)


def test_agent_error_is_base_exception():
    error = AgentError("Agent operation failed.")

    assert isinstance(error, Exception)
    assert str(error) == "Agent operation failed."


@pytest.mark.parametrize(
    "exception_type",
    [
        AgentConfigurationError,
        AgentTaskError,
        AgentStateError,
    ],
)
def test_direct_agent_exceptions_inherit_from_agent_error(
    exception_type,
):
    error = exception_type("Agent error.")

    assert isinstance(error, AgentError)
    assert isinstance(error, Exception)


@pytest.mark.parametrize(
    "exception_type",
    [
        AgentPlanningError,
        AgentExecutionError,
        AgentCancelledError,
    ],
)
def test_task_exceptions_inherit_from_agent_task_error(
    exception_type,
):
    error = exception_type("Agent task failed.")

    assert isinstance(error, AgentTaskError)
    assert isinstance(error, AgentError)
    assert isinstance(error, Exception)


def test_planning_error_has_correct_hierarchy():
    error = AgentPlanningError("Planning failed.")

    assert isinstance(error, AgentPlanningError)
    assert isinstance(error, AgentTaskError)
    assert isinstance(error, AgentError)


def test_execution_error_has_correct_hierarchy():
    error = AgentExecutionError("Execution failed.")

    assert isinstance(error, AgentExecutionError)
    assert isinstance(error, AgentTaskError)
    assert isinstance(error, AgentError)


def test_cancelled_error_has_correct_hierarchy():
    error = AgentCancelledError("Task cancelled.")

    assert isinstance(error, AgentCancelledError)
    assert isinstance(error, AgentTaskError)
    assert isinstance(error, AgentError)


def test_agent_exceptions_can_be_caught_by_base_class():
    with pytest.raises(AgentError):
        raise AgentPlanningError("Unable to create a plan.")


def test_task_exceptions_can_be_caught_by_task_error():
    with pytest.raises(AgentTaskError):
        raise AgentExecutionError("Step execution failed.")


def test_agent_state_error_is_not_a_task_error():
    error = AgentStateError("Invalid state transition.")

    assert isinstance(error, AgentError)
    assert not isinstance(error, AgentTaskError)


def test_exception_messages_are_preserved():
    message = "The task was cancelled."
    error = AgentCancelledError(message)

    assert str(error) == message