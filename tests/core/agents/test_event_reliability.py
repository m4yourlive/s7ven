
"""Tests for event publishing reliability and failure isolation."""

from uuid import uuid4

import pytest

from seven.core.agents.autonomous_agent import AutonomousAgent
from seven.core.agents.events import AgentTaskStarted
from seven.core.agents.interfaces import Executor, Planner
from seven.core.agents.models import AgentPlan, AgentStep, AgentTask
from seven.core.events import Event, EventBus


def test_event_bus_isolates_failing_subscribers():
    event_bus = EventBus()
    received_events: list[Event] = []

    def failing_handler(event: Event) -> None:
        raise RuntimeError("Subscriber failure")

    def successful_handler(event: Event) -> None:
        received_events.append(event)

    event_bus.subscribe(AgentTaskStarted, failing_handler)
    event_bus.subscribe(AgentTaskStarted, successful_handler)

    event = AgentTaskStarted(task_id=uuid4())
    published_count = event_bus.publish(event)

    assert published_count == 1
    assert received_events == [event]


def test_event_bus_continues_after_multiple_handler_failures():
    event_bus = EventBus()
    received_events: list[Event] = []

    def failing_handler(event: Event) -> None:
        raise RuntimeError("Handler failure")

    def successful_handler(event: Event) -> None:
        received_events.append(event)

    event_bus.subscribe(AgentTaskStarted, failing_handler)
    event_bus.subscribe(AgentTaskStarted, failing_handler)
    event_bus.subscribe(AgentTaskStarted, successful_handler)

    event = AgentTaskStarted(task_id=uuid4())
    published_count = event_bus.publish(event)

    assert published_count == 1
    assert received_events == [event]


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
        return "Completed"


@pytest.mark.asyncio
async def test_agent_execution_survives_event_handler_failure():
    event_bus = EventBus()
    received_events: list[Event] = []

    def failing_handler(event: Event) -> None:
        raise RuntimeError("Event handler failure")

    def successful_handler(event: Event) -> None:
        received_events.append(event)

    event_bus.subscribe(AgentTaskStarted, failing_handler)
    event_bus.subscribe(AgentTaskStarted, successful_handler)

    agent = AutonomousAgent(
        planner=FakePlanner(),
        executor=FakeExecutor(),
        event_bus=event_bus,
    )

    task = AgentTask(goal="Test event failure isolation")
    result = await agent.run(task)

    assert result.success is True
    assert result.output == "Completed"
    assert len(received_events) == 1
    assert received_events[0].task_id == task.task_id
