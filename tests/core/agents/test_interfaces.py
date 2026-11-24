"""Tests for autonomous agent interfaces."""

from collections.abc import Mapping
from uuid import UUID, uuid4

import pytest

from seven.core.agents.interfaces import Executor, Planner
from seven.core.agents.models import AgentPlan, AgentStep, AgentTask


def test_planner_is_abstract():
    with pytest.raises(TypeError):
        Planner()


def test_executor_is_abstract():
    with pytest.raises(TypeError):
        Executor()


def test_planner_can_be_implemented():
    class TestPlanner(Planner):
        async def plan(self, task: AgentTask) -> AgentPlan:
            step = AgentStep(
                instruction="Complete the task.",
                position=0,
            )
            return AgentPlan(
                task_id=task.task_id,
                steps=(step,),
            )

    planner = TestPlanner()

    assert isinstance(planner, Planner)


def test_executor_can_be_implemented():
    class TestExecutor(Executor):
        async def execute(
            self,
            task: AgentTask,
            step: AgentStep,
            context: Mapping[UUID, str],
        ) -> str:
            return f"Executed: {step.instruction}"

    executor = TestExecutor()

    assert isinstance(executor, Executor)


@pytest.mark.asyncio
async def test_planner_returns_agent_plan():
    class TestPlanner(Planner):
        async def plan(self, task: AgentTask) -> AgentPlan:
            step = AgentStep(
                instruction="Research the topic.",
                position=0,
            )
            return AgentPlan(
                task_id=task.task_id,
                steps=(step,),
            )

    task = AgentTask(goal="Research artificial intelligence.")
    planner = TestPlanner()

    plan = await planner.plan(task)

    assert isinstance(plan, AgentPlan)
    assert plan.task_id == task.task_id
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_executor_returns_step_output():
    class TestExecutor(Executor):
        async def execute(
            self,
            task: AgentTask,
            step: AgentStep,
            context: Mapping[UUID, str],
        ) -> str:
            previous_output = context.get(
                next(iter(context), UUID(int=0)),
                "",
            )
            return f"{step.instruction} {previous_output}".strip()

    task = AgentTask(goal="Complete a research task.")
    step = AgentStep(
        instruction="Analyze the findings.",
        position=1,
    )
    previous_step_id = uuid4()
    context = {
        previous_step_id: "Previous research completed."
    }

    executor = TestExecutor()
    output = await executor.execute(task, step, context)

    assert isinstance(output, str)
    assert "Analyze the findings." in output
    assert "Previous research completed." in output


@pytest.mark.asyncio
async def test_executor_receives_empty_context():
    class TestExecutor(Executor):
        async def execute(
            self,
            task: AgentTask,
            step: AgentStep,
            context: Mapping[UUID, str],
        ) -> str:
            return "Execution completed."

    task = AgentTask(goal="Run a simple task.")
    step = AgentStep(
        instruction="Execute the first step.",
        position=0,
    )

    executor = TestExecutor()
    output = await executor.execute(task, step, {})

    assert output == "Execution completed."


def test_interface_exports_are_available():
    from seven.core.agents.interfaces.executor import Executor as ExecutorModule
    from seven.core.agents.interfaces.planner import Planner as PlannerModule

    assert Executor is ExecutorModule
    assert Planner is PlannerModule