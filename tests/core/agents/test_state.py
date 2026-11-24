"""Tests for autonomous agent state management."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from seven.core.agents.models import (
    AgentPlan,
    AgentResult,
    AgentStep,
)
from seven.core.agents.state import AgentState, AgentStatus


def create_plan(task_id):
    first = AgentStep(
        instruction="Collect information.",
        position=0,
    )
    second = AgentStep(
        instruction="Analyze information.",
        position=1,
        dependencies=(first.step_id,),
    )

    return AgentPlan(
        task_id=task_id,
        steps=(first, second),
    )


def prepare_running_state():
    task_id = uuid4()
    plan = create_plan(task_id)
    state = AgentState(task_id=task_id)

    state.transition(AgentStatus.PLANNING)
    state.set_plan(plan)
    state.transition(AgentStatus.RUNNING)

    return state, plan


def test_agent_state_starts_pending():
    state = AgentState(task_id=uuid4())

    assert state.status == AgentStatus.PENDING
    assert state.is_active is False
    assert state.is_terminal is False
    assert state.plan is None
    assert state.result is None


def test_agent_state_rejects_invalid_task_id():
    with pytest.raises(TypeError):
        AgentState(task_id="invalid")


def test_agent_state_rejects_invalid_status():
    with pytest.raises(TypeError):
        AgentState(
            task_id=uuid4(),
            status="pending",
        )


def test_agent_state_rejects_plan_for_another_task():
    with pytest.raises(ValueError):
        AgentState(
            task_id=uuid4(),
            plan=create_plan(uuid4()),
        )


def test_agent_state_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        AgentState(
            task_id=uuid4(),
            updated_at=datetime.now(),
        )


def test_agent_state_transitions_to_planning():
    state = AgentState(task_id=uuid4())

    state.transition(AgentStatus.PLANNING)

    assert state.status == AgentStatus.PLANNING
    assert state.is_active is True


def test_agent_state_rejects_invalid_transition():
    state = AgentState(task_id=uuid4())

    with pytest.raises(ValueError):
        state.transition(AgentStatus.RUNNING)


def test_agent_state_sets_plan_and_becomes_ready():
    task_id = uuid4()
    plan = create_plan(task_id)
    state = AgentState(task_id=task_id)

    state.transition(AgentStatus.PLANNING)
    state.set_plan(plan)

    assert state.plan is plan
    assert state.status == AgentStatus.READY


def test_agent_state_rejects_plan_outside_planning():
    state = AgentState(task_id=uuid4())

    with pytest.raises(ValueError):
        state.set_plan(create_plan(state.task_id))


def test_agent_state_rejects_plan_for_another_task():
    state = AgentState(task_id=uuid4())
    state.transition(AgentStatus.PLANNING)

    with pytest.raises(ValueError):
        state.set_plan(create_plan(uuid4()))


def test_agent_state_starts_running():
    state, _ = prepare_running_state()

    assert state.status == AgentStatus.RUNNING
    assert state.started_at is not None
    assert state.is_active is True


def test_agent_state_starts_planned_step():
    state, plan = prepare_running_state()
    step = plan.steps[0]

    state.start_step(step.step_id)

    assert state.current_step_id == step.step_id


def test_agent_state_rejects_step_outside_plan():
    state, _ = prepare_running_state()

    with pytest.raises(ValueError):
        state.start_step(uuid4())


def test_agent_state_rejects_step_before_dependencies_complete():
    state, plan = prepare_running_state()
    dependent_step = plan.steps[1]

    with pytest.raises(ValueError):
        state.start_step(dependent_step.step_id)


def test_agent_state_rejects_concurrent_step_start():
    state, plan = prepare_running_state()

    state.start_step(plan.steps[0].step_id)

    with pytest.raises(ValueError):
        state.start_step(plan.steps[1].step_id)


def test_agent_state_completes_step():
    state, plan = prepare_running_state()
    step_id = plan.steps[0].step_id

    state.start_step(step_id)
    state.complete_step(step_id)

    assert state.completed_step_ids == (step_id,)
    assert state.current_step_id is None


def test_agent_state_rejects_wrong_step_completion():
    state, plan = prepare_running_state()

    state.start_step(plan.steps[0].step_id)

    with pytest.raises(ValueError):
        state.complete_step(plan.steps[1].step_id)


def test_agent_state_completes_successful_task():
    state, plan = prepare_running_state()

    for step in plan.steps:
        state.start_step(step.step_id)
        state.complete_step(step.step_id)

    result = AgentResult(
        task_id=state.task_id,
        success=True,
        output="Task completed.",
        completed_step_ids=state.completed_step_ids,
    )

    state.finish(result)

    assert state.status == AgentStatus.COMPLETED
    assert state.result is result
    assert state.completed_at is not None
    assert state.is_terminal is True
    assert state.is_active is False


def test_agent_state_finishes_failed_task():
    state, _ = prepare_running_state()

    result = AgentResult(
        task_id=state.task_id,
        success=False,
        error="Execution failed.",
    )

    state.finish(result)

    assert state.status == AgentStatus.FAILED
    assert state.error == "Execution failed."
    assert state.is_terminal is True


def test_agent_state_rejects_result_for_another_task():
    state, _ = prepare_running_state()

    result = AgentResult(
        task_id=uuid4(),
        success=False,
        error="Execution failed.",
    )

    with pytest.raises(ValueError):
        state.finish(result)


def test_agent_state_rejects_success_with_incomplete_steps():
    state, _ = prepare_running_state()

    result = AgentResult(
        task_id=state.task_id,
        success=True,
        output="Task completed.",
    )

    with pytest.raises(ValueError):
        state.finish(result)


def test_agent_state_rejects_completion_with_running_step():
    state, plan = prepare_running_state()
    state.start_step(plan.steps[0].step_id)

    result = AgentResult(
        task_id=state.task_id,
        success=True,
        output="Task completed.",
    )

    with pytest.raises(ValueError):
        state.finish(result)


def test_agent_state_can_be_cancelled():
    state = AgentState(task_id=uuid4())

    state.cancel()

    assert state.status == AgentStatus.CANCELLED
    assert state.completed_at is not None
    assert state.is_terminal is True


def test_agent_state_rejects_transition_after_completion():
    state = AgentState(task_id=uuid4())
    state.cancel()

    with pytest.raises(ValueError):
        state.transition(AgentStatus.PLANNING)


def test_agent_state_rejects_invalid_cancel_transition():
    state = AgentState(task_id=uuid4())
    state.cancel()

    with pytest.raises(ValueError):
        state.cancel()


def test_agent_state_updated_at_changes_after_transition():
    state = AgentState(task_id=uuid4())
    previous_updated_at = state.updated_at

    state.transition(AgentStatus.PLANNING)

    assert state.updated_at >= previous_updated_at
    assert state.updated_at.tzinfo == UTC