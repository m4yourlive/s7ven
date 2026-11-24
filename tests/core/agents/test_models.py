"""Tests for autonomous agent domain models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from seven.core.ai.models import AIMessage, MessageRole
from seven.core.agents.models import (
    AgentPlan,
    AgentResult,
    AgentStep,
    AgentTask,
)


def test_agent_task_creates_default_values():
    task = AgentTask(goal="Research a topic.")

    assert task.goal == "Research a topic."
    assert task.task_id is not None
    assert task.context == ()
    assert task.metadata == {}
    assert task.created_at.tzinfo is not None


def test_agent_task_rejects_empty_goal():
    with pytest.raises(ValueError):
        AgentTask(goal="   ")


def test_agent_task_rejects_invalid_context():
    with pytest.raises(TypeError):
        AgentTask(
            goal="Research a topic.",
            context=("invalid",),
        )


def test_agent_task_accepts_ai_message_context():
    message = AIMessage(
        role=MessageRole.USER,
        content="Previous conversation.",
    )

    task = AgentTask(
        goal="Continue the task.",
        context=(message,),
    )

    assert task.context == (message,)


def test_agent_task_metadata_is_immutable():
    metadata = {"source": "user"}
    task = AgentTask(
        goal="Research a topic.",
        metadata=metadata,
    )

    metadata["source"] = "system"

    assert task.metadata["source"] == "user"

    with pytest.raises(TypeError):
        task.metadata["source"] = "system"


def test_agent_step_creates_valid_step():
    step = AgentStep(
        instruction="Collect relevant information.",
        position=0,
    )

    assert step.instruction == "Collect relevant information."
    assert step.position == 0
    assert step.dependencies == ()


def test_agent_step_rejects_empty_instruction():
    with pytest.raises(ValueError):
        AgentStep(
            instruction=" ",
            position=0,
        )


def test_agent_step_rejects_negative_position():
    with pytest.raises(ValueError):
        AgentStep(
            instruction="Collect information.",
            position=-1,
        )


def test_agent_step_rejects_self_dependency():
    step_id = uuid4()

    with pytest.raises(ValueError):
        AgentStep(
            step_id=step_id,
            instruction="Execute a step.",
            position=0,
            dependencies=(step_id,),
        )


def test_agent_plan_accepts_valid_dependencies():
    first = AgentStep(
        instruction="Collect information.",
        position=0,
    )
    second = AgentStep(
        instruction="Analyze information.",
        position=1,
        dependencies=(first.step_id,),
    )

    plan = AgentPlan(
        task_id=uuid4(),
        steps=(first, second),
    )

    assert len(plan.steps) == 2
    assert plan.steps[1].dependencies == (first.step_id,)


def test_agent_plan_rejects_empty_steps():
    with pytest.raises(ValueError):
        AgentPlan(
            task_id=uuid4(),
            steps=(),
        )


def test_agent_plan_rejects_duplicate_step_ids():
    step_id = uuid4()

    first = AgentStep(
        step_id=step_id,
        instruction="First step.",
        position=0,
    )
    second = AgentStep(
        step_id=step_id,
        instruction="Second step.",
        position=1,
    )

    with pytest.raises(ValueError):
        AgentPlan(
            task_id=uuid4(),
            steps=(first, second),
        )


def test_agent_plan_rejects_missing_dependency():
    step = AgentStep(
        instruction="Analyze information.",
        position=1,
        dependencies=(uuid4(),),
    )

    with pytest.raises(ValueError):
        AgentPlan(
            task_id=uuid4(),
            steps=(step,),
        )


def test_agent_plan_rejects_dependency_on_later_step():
    first = AgentStep(
        instruction="First step.",
        position=0,
    )
    second = AgentStep(
        instruction="Second step.",
        position=1,
    )
    invalid_first = AgentStep(
        instruction="Invalid dependency.",
        position=0,
        dependencies=(second.step_id,),
    )

    with pytest.raises(ValueError):
        AgentPlan(
            task_id=uuid4(),
            steps=(invalid_first, second),
        )


def test_agent_result_accepts_success():
    task_id = uuid4()
    step_id = uuid4()

    result = AgentResult(
        task_id=task_id,
        success=True,
        output="Task completed.",
        completed_step_ids=(step_id,),
    )

    assert result.success is True
    assert result.output == "Task completed."
    assert result.error is None


def test_agent_result_accepts_failure():
    result = AgentResult(
        task_id=uuid4(),
        success=False,
        error="Execution failed.",
    )

    assert result.success is False
    assert result.error == "Execution failed."


def test_agent_result_rejects_failure_without_error():
    with pytest.raises(ValueError):
        AgentResult(
            task_id=uuid4(),
            success=False,
        )


def test_agent_result_rejects_success_with_error():
    with pytest.raises(ValueError):
        AgentResult(
            task_id=uuid4(),
            success=True,
            output="Completed.",
            error="Unexpected error.",
        )


def test_agent_result_rejects_duplicate_completed_steps():
    step_id = uuid4()

    with pytest.raises(ValueError):
        AgentResult(
            task_id=uuid4(),
            success=True,
            completed_step_ids=(step_id, step_id),
        )


def test_agent_models_require_timezone_aware_timestamps():
    with pytest.raises(ValueError):
        AgentTask(
            goal="Research a topic.",
            created_at=datetime.now(),
        )