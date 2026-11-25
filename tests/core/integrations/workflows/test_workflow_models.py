
"""Tests for workflow domain models."""

from uuid import uuid4

import pytest

from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepResult,
    WorkflowStepStatus,
)


def test_workflow_step_accepts_valid_data():
    step = WorkflowStep(
        name="Open application",
        action="application.launch",
        parameters={"application": "notepad"},
    )

    assert step.name == "Open application"
    assert step.action == "application.launch"
    assert step.parameters == {"application": "notepad"}
    assert step.step_id is not None


def test_workflow_step_normalizes_strings():
    step = WorkflowStep(
        name="  Open application  ",
        action="  application.launch  ",
    )

    assert step.name == "Open application"
    assert step.action == "application.launch"


@pytest.mark.parametrize("name", ["", "   ", 123])
def test_workflow_step_rejects_invalid_name(name):
    expected_error = TypeError if not isinstance(name, str) else ValueError

    with pytest.raises(expected_error):
        WorkflowStep(name=name, action="application.launch")


@pytest.mark.parametrize("action", ["", "   ", 123])
def test_workflow_step_rejects_invalid_action(action):
    expected_error = (
        TypeError if not isinstance(action, str) else ValueError
    )

    with pytest.raises(expected_error):
        WorkflowStep(name="Launch", action=action)


def test_workflow_step_rejects_invalid_parameters():
    with pytest.raises(TypeError):
        WorkflowStep(
            name="Launch",
            action="application.launch",
            parameters=[],
        )


def test_workflow_step_rejects_invalid_id():
    with pytest.raises(TypeError):
        WorkflowStep(
            name="Launch",
            action="application.launch",
            step_id="invalid",
        )


def test_workflow_step_copies_parameters():
    parameters = {"application": "notepad"}

    step = WorkflowStep(
        name="Launch",
        action="application.launch",
        parameters=parameters,
    )

    parameters["application"] = "calculator"

    assert step.parameters["application"] == "notepad"


def test_workflow_step_is_immutable():
    step = WorkflowStep(
        name="Launch",
        action="application.launch",
    )

    with pytest.raises(AttributeError):
        step.name = "Changed"


def test_workflow_accepts_valid_data():
    step = WorkflowStep(
        name="Launch",
        action="application.launch",
    )

    workflow = Workflow(
        name="Desktop startup",
        steps=(step,),
        description="Launch desktop applications.",
    )

    assert workflow.name == "Desktop startup"
    assert workflow.steps == (step,)
    assert workflow.description == "Launch desktop applications."
    assert workflow.workflow_id is not None


def test_workflow_normalizes_name():
    workflow = Workflow(name="  Startup  ", steps=())

    assert workflow.name == "Startup"


@pytest.mark.parametrize("name", ["", "   ", 123])
def test_workflow_rejects_invalid_name(name):
    expected_error = TypeError if not isinstance(name, str) else ValueError

    with pytest.raises(expected_error):
        Workflow(name=name, steps=())


def test_workflow_rejects_non_tuple_steps():
    with pytest.raises(TypeError):
        Workflow(
            name="Startup",
            steps=[WorkflowStep(name="Launch", action="launch")],
        )


def test_workflow_rejects_invalid_step():
    with pytest.raises(TypeError):
        Workflow(name="Startup", steps=("invalid",))


def test_workflow_rejects_duplicate_step_ids():
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

    with pytest.raises(ValueError):
        Workflow(name="Duplicate", steps=(first, second))


def test_workflow_rejects_invalid_id():
    with pytest.raises(TypeError):
        Workflow(
            name="Startup",
            steps=(),
            workflow_id="invalid",
        )


def test_workflow_rejects_invalid_description():
    with pytest.raises(TypeError):
        Workflow(
            name="Startup",
            steps=(),
            description=123,
        )


def test_workflow_step_result_accepts_valid_data():
    step_id = uuid4()

    result = WorkflowStepResult(
        step_id=step_id,
        status=WorkflowStepStatus.COMPLETED,
        output={"pid": 1234},
    )

    assert result.step_id == step_id
    assert result.status is WorkflowStepStatus.COMPLETED
    assert result.output == {"pid": 1234}


def test_workflow_step_result_rejects_invalid_id():
    with pytest.raises(TypeError):
        WorkflowStepResult(
            step_id="invalid",
            status=WorkflowStepStatus.COMPLETED,
        )


def test_workflow_step_result_rejects_invalid_status():
    with pytest.raises(TypeError):
        WorkflowStepResult(
            step_id=uuid4(),
            status="completed",
        )


def test_workflow_step_result_rejects_invalid_error():
    with pytest.raises(TypeError):
        WorkflowStepResult(
            step_id=uuid4(),
            status=WorkflowStepStatus.FAILED,
            error=123,
        )


def test_workflow_execution_accepts_valid_data():
    workflow_id = uuid4()
    result = WorkflowStepResult(
        step_id=uuid4(),
        status=WorkflowStepStatus.COMPLETED,
    )

    execution = WorkflowExecution(
        workflow_id=workflow_id,
        status=WorkflowStatus.COMPLETED,
        step_results=(result,),
    )

    assert execution.workflow_id == workflow_id
    assert execution.status is WorkflowStatus.COMPLETED
    assert execution.step_results == (result,)
    assert execution.execution_id is not None


def test_workflow_execution_uses_pending_status_by_default():
    execution = WorkflowExecution(workflow_id=uuid4())

    assert execution.status is WorkflowStatus.PENDING
    assert execution.step_results == ()
    assert execution.error is None


def test_workflow_execution_rejects_invalid_workflow_id():
    with pytest.raises(TypeError):
        WorkflowExecution(workflow_id="invalid")


def test_workflow_execution_rejects_invalid_execution_id():
    with pytest.raises(TypeError):
        WorkflowExecution(
            workflow_id=uuid4(),
            execution_id="invalid",
        )


def test_workflow_execution_rejects_invalid_status():
    with pytest.raises(TypeError):
        WorkflowExecution(
            workflow_id=uuid4(),
            status="completed",
        )


def test_workflow_execution_rejects_non_tuple_results():
    with pytest.raises(TypeError):
        WorkflowExecution(
            workflow_id=uuid4(),
            step_results=[],
        )


def test_workflow_execution_rejects_invalid_step_result():
    with pytest.raises(TypeError):
        WorkflowExecution(
            workflow_id=uuid4(),
            step_results=("invalid",),
        )


def test_workflow_execution_rejects_invalid_error():
    with pytest.raises(TypeError):
        WorkflowExecution(
            workflow_id=uuid4(),
            error=123,
        )


def test_workflow_execution_is_immutable():
    execution = WorkflowExecution(workflow_id=uuid4())

    with pytest.raises(AttributeError):
        execution.status = WorkflowStatus.COMPLETED
