
"""Tests for autonomous agent event integration."""

from uuid import uuid4

import pytest

from seven.core.agents.autonomous_agent import AutonomousAgent
from seven.core.agents.events import (
    AgentPlanCreated,
    AgentPlanningStarted,
    AgentStepCompleted,
    AgentStepStarted,
    AgentTaskCompleted,
    AgentTaskStarted,
)
from seven.core.agents.interfaces import Executor, Planner
from seven.core.agents.models import (
    AgentPlan,
    AgentResult,
    AgentStep,
    AgentTask,
)
from seven.core.events import Event, EventBus


class FakePlanner(Planner):
    async def plan(self, task: AgentTask) -> AgentPlan:
        return AgentPlan(
            task_id=task.task_id,
            steps=(
                AgentStep(
                    instruction="Complete the task",
                    position=0,
                ),
            ),
        )


class FakeExecutor(Executor):
    async def execute(
        self,
        task: AgentTask,
        step: AgentStep,
        context,
    ) -> str:
        return "Task completed"


@pytest.mark.asyncio
async def test_agent_publishes_complete_lifecycle():
    event_bus = EventBus()
    received_events: list[Event] = []

    event_types = (
        AgentTaskStarted,
        AgentPlanningStarted,
        AgentPlanCreated,
        AgentStepStarted,
        AgentStepCompleted,
        AgentTaskCompleted,
    )

    for event_type in event_types:
        event_bus.subscribe(
            event_type,
            received_events.append,
        )

    agent = AutonomousAgent(
        planner=FakePlanner(),
        executor=FakeExecutor(),
        event_bus=event_bus,
    )

    task = AgentTask(goal="Complete a test task")
    result = await agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.success is True

    assert [type(event) for event in received_events] == list(
        event_types
    )

    assert all(
        event.task_id == task.task_id
        for event in received_events
    )


@pytest.mark.asyncio
async def test_agent_runs_without_event_bus():
    agent = AutonomousAgent(
        planner=FakePlanner(),
        executor=FakeExecutor(),
    )

    task = AgentTask(goal="Run without event publishing")
    result = await agent.run(task)

    assert result.success is True
    assert result.output == "Task completed"
    assert agent.event_bus is None


@pytest.mark.asyncio
async def test_agent_publishes_failure_event():
    event_bus = EventBus()
    received_events: list[Event] = []

    from seven.core.agents.events import AgentTaskFailed

    event_bus.subscribe(
        AgentTaskFailed,
        received_events.append,
    )

    class FailingPlanner(Planner):
        async def plan(self, task: AgentTask) -> AgentPlan:
            raise RuntimeError("Planning failed")

    agent = AutonomousAgent(
        planner=FailingPlanner(),
        executor=FakeExecutor(),
        event_bus=event_bus,
    )

    task = AgentTask(goal="Trigger planning failure")
    result = await agent.run(task)

    assert result.success is False
    assert len(received_events) == 1
    assert received_events[0].task_id == task.task_id
    assert received_events[0].result is result
