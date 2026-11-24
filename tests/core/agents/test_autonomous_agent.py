
"""Tests for the autonomous agent orchestration service."""

from uuid import uuid4

import pytest

from seven.core.agents.autonomous_agent import AutonomousAgent
from seven.core.agents.exceptions import AgentTaskError
from seven.core.agents.interfaces import Executor, Planner
from seven.core.agents.models import (
    AgentPlan,
    AgentResult,
    AgentStep,
    AgentTask,
)
from seven.core.agents.state import AgentStatus


class StubPlanner(Planner):
    def __init__(self, plan=None, error=None):
        self.plan_result = plan
        self.error = error
        self.calls = []

    async def plan(self, task):
        self.calls.append(task)
        if self.error is not None:
            raise self.error
        return self.plan_result


class StubExecutor(Executor):
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs or {}
        self.error = error
        self.calls = []

    async def execute(self, task, step, context):
        self.calls.append((task, step, dict(context)))
        if self.error is not None:
            raise self.error
        return self.outputs.get(step.step_id, f"Output: {step.instruction}")


def create_task():
    return AgentTask(goal="Complete a sample task.")


def create_plan(task, steps=None):
    if steps is None:
        steps = (
            AgentStep(instruction="First step", position=0),
            AgentStep(instruction="Second step", position=1),
        )

    return AgentPlan(task_id=task.task_id, steps=steps)


@pytest.mark.asyncio
async def test_run_completes_all_steps():
    task = create_task()
    plan = create_plan(task)
    planner = StubPlanner(plan=plan)
    executor = StubExecutor()
    agent = AutonomousAgent(planner, executor)

    result = await agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert len(result.completed_step_ids) == 2
    assert result.output == "Output: Second step"
    assert len(planner.calls) == 1
    assert len(executor.calls) == 2

    state = agent.get_state(task.task_id)
    assert state is not None
    assert state.status is AgentStatus.COMPLETED
    assert state.is_terminal is True
    assert agent.active_count == 0


@pytest.mark.asyncio
async def test_run_passes_previous_outputs_as_context():
    task = create_task()
    plan = create_plan(task)
    first_step, second_step = plan.steps

    executor = StubExecutor(
        outputs={
            first_step.step_id: "First output",
            second_step.step_id: "Second output",
        }
    )
    agent = AutonomousAgent(StubPlanner(plan=plan), executor)

    await agent.run(task)

    second_call_context = executor.calls[1][2]

    assert second_call_context == {
        first_step.step_id: "First output",
    }


@pytest.mark.asyncio
async def test_run_rejects_plan_for_different_task():
    task = create_task()
    other_task = create_task()
    planner = StubPlanner(plan=create_plan(other_task))
    agent = AutonomousAgent(planner, StubExecutor())

    result = await agent.run(task)

    assert result.success is False
    assert "different task" in result.error
    assert agent.get_state(task.task_id).status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_returns_failure_when_planning_fails():
    task = create_task()
    planner = StubPlanner(error=RuntimeError("Planning unavailable"))
    agent = AutonomousAgent(planner, StubExecutor())

    result = await agent.run(task)

    assert result.success is False
    assert "Task planning failed" in result.error
    assert result.completed_step_ids == ()
    assert agent.get_state(task.task_id).status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_returns_failure_when_step_execution_fails():
    task = create_task()
    plan = create_plan(task)
    first_step = plan.steps[0]
    executor = StubExecutor(
        outputs={},
        error=RuntimeError("Execution unavailable"),
    )
    agent = AutonomousAgent(StubPlanner(plan=plan), executor)

    result = await agent.run(task)

    assert result.success is False
    assert "Task execution failed" in result.error
    assert result.failed_step_id == first_step.step_id
    assert result.completed_step_ids == ()
    assert agent.get_state(task.task_id).status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_rejects_duplicate_task_id():
    task = create_task()
    plan = create_plan(task)
    agent = AutonomousAgent(StubPlanner(plan=plan), StubExecutor())

    await agent.run(task)

    with pytest.raises(AgentTaskError):
        await agent.run(task)


def test_constructor_validates_dependencies():
    with pytest.raises(TypeError):
        AutonomousAgent(object(), StubExecutor())

    with pytest.raises(TypeError):
        AutonomousAgent(StubPlanner(), object())


def test_get_state_returns_none_for_unknown_task():
    agent = AutonomousAgent(StubPlanner(), StubExecutor())

    assert agent.get_state(uuid4()) is None


def test_active_count_starts_at_zero():
    agent = AutonomousAgent(StubPlanner(), StubExecutor())

    assert agent.active_count == 0
