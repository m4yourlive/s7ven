
"""Tests for workflow definition validation."""

import pytest

from seven.core.integrations.workflows.exceptions import (
    WorkflowValidationError,
)
from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowStep,
)
from seven.core.integrations.workflows.validation import (
    DefaultWorkflowValidator,
)


@pytest.fixture
def validator():
    return DefaultWorkflowValidator()


def test_validator_accepts_valid_workflow(validator):
    workflow = Workflow(
        name="Desktop startup",
        steps=(
            WorkflowStep(
                name="Launch application",
                action="application.launch",
                parameters={"application": "notepad"},
            ),
        ),
    )

    validator.validate(workflow)


def test_validator_rejects_non_workflow_object(validator):
    with pytest.raises(WorkflowValidationError):
        validator.validate("invalid")


def test_validator_rejects_empty_workflow(validator):
    workflow = Workflow(name="Empty workflow", steps=())

    with pytest.raises(WorkflowValidationError, match="at least one step"):
        validator.validate(workflow)


def test_validator_rejects_empty_step_name(validator):
    step = WorkflowStep.__new__(WorkflowStep)
    object.__setattr__(step, "name", "")
    object.__setattr__(step, "action", "application.launch")
    object.__setattr__(step, "parameters", {})
    object.__setattr__(step, "step_id", __import__("uuid").uuid4())

    workflow = Workflow(name="Invalid workflow", steps=(step,))

    with pytest.raises(WorkflowValidationError, match="empty name"):
        validator.validate(workflow)


def test_validator_rejects_empty_step_action(validator):
    step = WorkflowStep.__new__(WorkflowStep)
    object.__setattr__(step, "name", "Launch")
    object.__setattr__(step, "action", "")
    object.__setattr__(step, "parameters", {})
    object.__setattr__(step, "step_id", __import__("uuid").uuid4())

    workflow = Workflow(name="Invalid workflow", steps=(step,))

    with pytest.raises(WorkflowValidationError, match="empty action"):
        validator.validate(workflow)



def test_validator_rejects_duplicate_step_ids(validator):
    from uuid import uuid4

    step_id = uuid4()

    first = WorkflowStep(
        name="First",
        action="first",
        step_id=step_id,
    )
    second = WorkflowStep(
        name="Second",
        action="second",
        step_id=step_id,
    )

    workflow = Workflow(
        name="Duplicate workflow",
        steps=(first,),
    )

    object.__setattr__(workflow, "steps", (first, second))

    with pytest.raises(WorkflowValidationError, match="Duplicate step ID"):
        validator.validate(workflow)


def test_validator_rejects_invalid_step_parameters(validator):
    step = WorkflowStep.__new__(WorkflowStep)
    object.__setattr__(step, "name", "Launch")
    object.__setattr__(step, "action", "application.launch")
    object.__setattr__(step, "parameters", None)
    object.__setattr__(step, "step_id", __import__("uuid").uuid4())

    workflow = Workflow(name="Invalid workflow", steps=(step,))

    with pytest.raises(WorkflowValidationError, match="parameters"):
        validator.validate(workflow)
