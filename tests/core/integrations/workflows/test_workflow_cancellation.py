
"""Tests for workflow cancellation and error handling."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from seven.core.integrations.workflows.engine import WorkflowEngine
from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepStatus,
)


@pytest.fixture
def validator():
    class Validator:
        def validate(self, workflow):
            if not isinstance(workflow, Workflow):
                raise ValueError("Invalid workflow.")

    return Validator()


@pytest.fixture
def workflow():
    return Workflow(
        name="Cancellation workflow",
        steps=(
            WorkflowStep(name="First", action="first"),
            WorkflowStep(name="Second", action="second"),
            WorkflowStep(name="Third", action="third"),
        ),
    )


@pytest.mark.asyncio
async def test_engine_cancels_before_execution(validator, workflow):
    handler = AsyncMock()
    cancellation_event = asyncio.Event()
    cancellation_event.set()

    engine = WorkflowEngine(handler, validator)

    result = await engine.execute(
        workflow,
        cancellation_event=cancellation_event,
    )

    assert result.status is WorkflowStatus.CANCELLED
    assert len(result.step_results) == 3
    assert all(
        item.status is WorkflowStepStatus.SKIPPED
        for item in result.step_results
    )
    handler.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_cancels_after_current_step(validator, workflow):
    handler = AsyncMock()
    cancellation_event = asyncio.Event()

    async def execute_action(action, parameters):
        cancellation_event.set()
        return "completed"

    handler.execute.side_effect = execute_action

    engine = WorkflowEngine(handler, validator)

    result = await engine.execute(
        workflow,
        cancellation_event=cancellation_event,
    )

    assert result.status is WorkflowStatus.CANCELLED
    assert result.step_results[0].status is WorkflowStepStatus.COMPLETED
    assert result.step_results[1].status is WorkflowStepStatus.SKIPPED
    assert result.step_results[2].status is WorkflowStepStatus.SKIPPED
    assert handler.execute.await_count == 1


@pytest.mark.asyncio
async def test_engine_preserves_failed_step_error(validator, workflow):
    handler = AsyncMock()
    handler.execute.side_effect = RuntimeError("Application unavailable")

    engine = WorkflowEngine(handler, validator)
    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.FAILED
    assert result.error == "Application unavailable"
    assert result.step_results[0].status is WorkflowStepStatus.FAILED
    assert all(
        item.status is WorkflowStepStatus.SKIPPED
        for item in result.step_results[1:]
    )
    assert handler.execute.await_count == 1


@pytest.mark.asyncio
async def test_engine_propagates_external_task_cancellation(
    validator,
    workflow,
):
    handler = AsyncMock()

    async def execute_action(action, parameters):
        raise asyncio.CancelledError

    handler.execute.side_effect = execute_action

    store = AsyncMock()
    engine = WorkflowEngine(handler, validator, store)

    with pytest.raises(asyncio.CancelledError):
        await engine.execute(workflow)

    store.save.assert_awaited_once()
    saved_execution = store.save.await_args.args[0]

    assert saved_execution.status is WorkflowStatus.CANCELLED
    assert saved_execution.error == "Workflow task was cancelled."


@pytest.mark.asyncio
async def test_engine_saves_cancelled_execution(validator, workflow):
    handler = AsyncMock()
    cancellation_event = asyncio.Event()

    async def execute_action(action, parameters):
        cancellation_event.set()
        return "completed"

    handler.execute.side_effect = execute_action

    store = AsyncMock()
    engine = WorkflowEngine(handler, validator, store)

    result = await engine.execute(
        workflow,
        cancellation_event=cancellation_event,
    )

    assert result.status is WorkflowStatus.CANCELLED
    store.save.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_engine_continues_after_successful_independent_steps(
    validator,
):
    workflow = Workflow(
        name="Independent workflow",
        steps=(
            WorkflowStep(name="First", action="first"),
            WorkflowStep(name="Second", action="second"),
        ),
    )

    handler = AsyncMock()
    handler.execute.side_effect = ["one", "two"]

    engine = WorkflowEngine(handler, validator)
    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.COMPLETED
    assert handler.execute.await_count == 2
