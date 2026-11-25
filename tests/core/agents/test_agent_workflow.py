
"""End-to-end tests for autonomous agent workflows."""

import pytest

from seven.core.agents.autonomous_agent import AutonomousAgent
from seven.core.agents.interfaces import Planner, Executor
from seven.core.agents.models import AgentPlan, AgentStep, AgentTask
from seven.core.agents.state import AgentStatus


class WorkflowPlanner(Planner):
    def __init__(self, steps):
        self.steps = steps
        self.calls = []

    async def plan(self, task):
        self.calls.append(task)
        return AgentPlan(task_id=task.task_id, steps=self.steps)


class WorkflowExecutor(Executor):
    def __init__(self, outputs=None, fail_on_step=None):
        self.outputs = outputs or {}
        self.fail_on_step = fail_on_step
        self.calls = []

    async def execute(self, task, step, context):
        self.calls.append({
            "task": task,
            "step": step,
            "context": dict(context),
        })

        if step.step_id == self.fail_on_step:
            raise RuntimeError("Simulated execution failure.")

        return self.outputs.get(
            step.step_id,
            f"Completed: {step.instruction}",
        )


@pytest.mark.asyncio
async def test_complete_workflow_propagates_context_across_all_steps():
    task = AgentTask(goal="Complete a multi-step workflow.")

    steps = (
        AgentStep(instruction="Research the topic", position=0),
        AgentStep(instruction="Analyze the findings", position=1),
        AgentStep(instruction="Prepare the final result", position=2),
    )

    outputs = {
        steps[0].step_id: "Research findings",
        steps[1].step_id: "Analysis results",
        steps[2].step_id: "Final report",
    }

    planner = WorkflowPlanner(steps)
    executor = WorkflowExecutor(outputs=outputs)
    agent = AutonomousAgent(planner, executor)

    result = await agent.run(task)

    assert result.success is True
    assert result.output == "Final report"
    assert result.completed_step_ids == tuple(
        step.step_id for step in steps
    )

    assert len(planner.calls) == 1
    assert len(executor.calls) == 3

    assert executor.calls[0]["context"] == {}
    assert executor.calls[1]["context"] == {
        steps[0].step_id: "Research findings",
    }
    assert executor.calls[2]["context"] == {
        steps[0].step_id: "Research findings",
        steps[1].step_id: "Analysis results",
    }

    state = agent.get_state(task.task_id)

    assert state is not None
    assert state.status is AgentStatus.COMPLETED
    assert state.completed_step_ids == tuple(
        step.step_id for step in steps
    )
    assert agent.active_count == 0


@pytest.mark.asyncio
async def test_workflow_preserves_completed_steps_when_later_step_fails():
    task = AgentTask(goal="Complete a workflow with a failing step.")

    steps = (
        AgentStep(instruction="Collect information", position=0),
        AgentStep(instruction="Process information", position=1),
        AgentStep(instruction="Generate report", position=2),
    )

    planner = WorkflowPlanner(steps)
    executor = WorkflowExecutor(
        outputs={steps[0].step_id: "Collected data"},
        fail_on_step=steps[1].step_id,
    )

    agent = AutonomousAgent(planner, executor)

    result = await agent.run(task)

    assert result.success is False
    assert result.failed_step_id == steps[1].step_id
    assert result.completed_step_ids == (steps[0].step_id,)
    assert len(executor.calls) == 2

    assert executor.calls[1]["context"] == {
        steps[0].step_id: "Collected data",
    }

    state = agent.get_state(task.task_id)

    assert state is not None
    assert state.status is AgentStatus.FAILED
    assert state.completed_step_ids == (steps[0].step_id,)
    assert state.current_step_id is None
    assert agent.active_count == 0
