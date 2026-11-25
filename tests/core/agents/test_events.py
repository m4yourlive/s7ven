
"""Tests for autonomous agent event models."""

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from seven.core.agents.events import (
    AgentPlanCreated,
    AgentPlanningStarted,
    AgentStepCompleted,
    AgentStepStarted,
    AgentTaskCancelled,
    AgentTaskCompleted,
    AgentTaskFailed,
    AgentTaskStarted,
)
from seven.core.agents.models import (
    AgentPlan,
    AgentResult,
    AgentStep,
)
from seven.core.events import Event


def test_agent_task_started_event():
    task_id = uuid4()
    event = AgentTaskStarted(task_id=task_id)

    assert isinstance(event, Event)
    assert event.task_id == task_id
    assert event.event_id is not None


def test_agent_planning_started_event():
    task_id = uuid4()
    event = AgentPlanningStarted(task_id=task_id)

    assert event.task_id == task_id


def test_agent_plan_created_event():
    task_id = uuid4()
    plan = AgentPlan(
        task_id=task_id,
        steps=(
            AgentStep(instruction="Complete task", position=0),
        ),
    )

    event = AgentPlanCreated(task_id=task_id, plan=plan)

    assert event.task_id == task_id
    assert event.plan is plan


def test_agent_step_started_event():
    task_id = uuid4()
    step = AgentStep(instruction="Execute step", position=0)

    event = AgentStepStarted(task_id=task_id, step=step)

    assert event.task_id == task_id
    assert event.step is step


def test_agent_step_completed_event():
    task_id = uuid4()
    step = AgentStep(instruction="Execute step", position=0)

    event = AgentStepCompleted(
        task_id=task_id,
        step=step,
        output="Step completed",
    )

    assert event.task_id == task_id
    assert event.step is step
    assert event.output == "Step completed"


def test_agent_task_completed_event():
    task_id = uuid4()
    result = AgentResult(
        task_id=task_id,
        success=True,
        output="Completed",
    )

    event = AgentTaskCompleted(task_id=task_id, result=result)

    assert event.task_id == task_id
    assert event.result is result


def test_agent_task_failed_event():
    task_id = uuid4()
    result = AgentResult(
        task_id=task_id,
        success=False,
        error="Execution failed",
    )

    event = AgentTaskFailed(task_id=task_id, result=result)

    assert event.task_id == task_id
    assert event.result is result


def test_agent_task_cancelled_event():
    task_id = uuid4()
    event = AgentTaskCancelled(task_id=task_id)

    assert event.task_id == task_id


def test_agent_events_are_immutable():
    event = AgentTaskStarted(task_id=uuid4())

    with pytest.raises(FrozenInstanceError):
        event.task_id = uuid4()


def test_agent_events_have_unique_ids():
    first = AgentTaskStarted(task_id=uuid4())
    second = AgentTaskStarted(task_id=uuid4())

    assert first.event_id != second.event_id
