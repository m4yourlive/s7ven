
"""Tests for workflow step dependencies."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from seven.core.integrations.workflows.engine import WorkflowEngine
from seven.core.integrations.workflows.exceptions import (
    WorkflowValidationError,
)
from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepStatus,
)
from seven.core.integrations.workflows.validation import (
    DefaultWorkflowValidator,
)


@pytest.fixture
def validator():
    return DefaultWorkflowValidator()


@pytest.mark.asyncio
async def test_engine_respects_step_dependencies(validator):
    first_id = uuid4()
    second_id = uuid4()

    first = WorkflowStep(
        name="First",
        action="first",
        step_id=first_id,
    )
    second = WorkflowStep(
        name="Second",
        action="second",
        step_id=second_id,
        dependencies=(first_id,),
    )

    workflow = Workflow(
        name="Dependency workflow",
        steps=(second, first),
    )

    handler = AsyncMock()
    handler.execute.side_effect = ["first output", "second output"]

    engine = WorkflowEngine(handler, validator)
    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.COMPLETED
    assert handler.execute.await_args_list[0].args[0] == "first"
    assert handler.execute.await_args_list[1].args[0] == "second"


@pytest.mark.asyncio
async def test_engine_skips_dependent_steps_after_failure(validator):
    first_id = uuid4()

    first = WorkflowStep(
        name="First",
        action="first",
        step_id=first_id,
    )
    second = WorkflowStep(
        name="Second",
        action="second",
        dependencies=(first_id,),
    )

    workflow = Workflow(
        name="Failure workflow",
        steps=(first, second),
    )

    handler = AsyncMock()
    handler.execute.side_effect = RuntimeError("Action failed")

    engine = WorkflowEngine(handler, validator)
    result = await engine.execute(workflow)

    assert result.status is WorkflowStatus.FAILED
    assert result.step_results[0].status is WorkflowStepStatus.FAILED
    assert result.step_results[1].status is WorkflowStepStatus.SKIPPED
    assert handler.execute.await_count == 1


def test_validator_rejects_unknown_dependency(validator):
    step = WorkflowStep(
        name="Dependent",
        action="dependent",
        dependencies=(uuid4(),),
    )
    workflow = Workflow(name="Invalid dependency", steps=(step,))

    with pytest.raises(WorkflowValidationError, match="unknown dependency"):
        validator.validate(workflow)


def test_validator_rejects_cyclic_dependencies(validator):
    first_id = uuid4()
    second_id = uuid4()

    first = WorkflowStep(
        name="First",
        action="first",
        step_id=first_id,
        dependencies=(second_id,),
    )
    second = WorkflowStep(
        name="Second",
        action="second",
        step_id=second_id,
        dependencies=(first_id,),
    )

    workflow = Workflow(
        name="Cyclic workflow",
        steps=(first, second),
    )

    with pytest.raises(WorkflowValidationError, match="cyclic"):
        validator.validate(workflow)


def test_validator_accepts_independent_steps(validator):
    workflow = Workflow(
        name="Independent workflow",
        steps=(
            WorkflowStep(name="First", action="first"),
            WorkflowStep(name="Second", action="second"),
        ),
    )

    validator.validate(workflow)
