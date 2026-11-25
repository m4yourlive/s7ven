
"""Tests for the workflow execution engine."""

from unittest.mock import AsyncMock

import pytest

from seven.core.integrations.workflows.engine import WorkflowEngine
from seven.core.integrations.workflows.exceptions import (
    WorkflowExecutionError,
)
from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepStatus,
)


@pytest.fixture
def workflow():
    return Workflow(
        name="Test workflow",
        steps=(
            WorkflowStep(
                name="First step",
                action="test.first",
                parameters={"value": 1},
            ),
            WorkflowStep(
                name="Second step",
                action="test.second",
                parameters={"value": 2},
            ),
        ),
    )


@pytest.fixture
def validator():
    class Validator:
        def validate(self, workflow):
            if not isinstance(workflow, Workflow):
                raise ValueError("Invalid workflow.")

    return Validator()


@pytest.fixture
def action_handler():
    handler = AsyncMock()
    handler.execute.side_effect = ["first output", "second output"]
    return handler


@pytest.fixture
def engine(action_handler, validator):
    return WorkflowEngine(
        action_handler=action_handler,
        validator=validator,
    )


@pytest.mark.asyncio
async def test_engine_executes_all_steps(engine, workflow, action_handler):
    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.COMPLETED
    assert len(result.step_results) == 2
    assert all(
        item.status is WorkflowStepStatus.COMPLETED
        for item in result.step_results
    )
    assert action_handler.execute.await_count == 2


@pytest.mark.asyncio
async def test_engine_preserves_step_outputs(engine, workflow):
    result = await engine.execute(workflow)

    assert result.step_results[0].output == "first output"
    assert result.step_results[1].output == "second output"


@pytest.mark.asyncio
async def test_engine_stops_after_failed_step(
    workflow,
    validator,
):
    handler = AsyncMock()
    handler.execute.side_effect = [
        "first output",
        RuntimeError("Action failed"),
    ]

    engine = WorkflowEngine(handler, validator)
    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.FAILED
    assert len(result.step_results) == 2
    assert result.step_results[-1].status is WorkflowStepStatus.FAILED
    assert result.error == "Action failed"
    assert handler.execute.await_count == 2


@pytest.mark.asyncio
async def test_engine_calls_validator_before_execution(
    workflow,
    action_handler,
):
    validator = AsyncMock()
    engine = WorkflowEngine(action_handler, validator)

    await engine.execute(workflow)

    validator.validate.assert_called_once_with(workflow)


@pytest.mark.asyncio
async def test_engine_does_not_execute_when_validation_fails(
    workflow,
    action_handler,
):
    validator = AsyncMock()
    validator.validate.side_effect = ValueError("Invalid workflow")

    engine = WorkflowEngine(action_handler, validator)

    with pytest.raises(ValueError, match="Invalid workflow"):
        await engine.execute(workflow)

    action_handler.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_saves_completed_execution(workflow, validator):
    handler = AsyncMock()
    handler.execute.side_effect = ["first", "second"]

    store = AsyncMock()
    engine = WorkflowEngine(handler, validator, store)

    result = await engine.execute(workflow)

    store.save.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_engine_saves_failed_execution(workflow, validator):
    handler = AsyncMock()
    handler.execute.side_effect = [
        "first",
        RuntimeError("Action failed"),
    ]

    store = AsyncMock()
    engine = WorkflowEngine(handler, validator, store)

    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.FAILED
    store.save.assert_awaited_once_with(result)


def test_engine_rejects_missing_dependencies():
    with pytest.raises(ValueError):
        WorkflowEngine(None, object())

    with pytest.raises(ValueError):
        WorkflowEngine(object(), None)
