
"""Tests for agent event identity and timestamp metadata."""

from dataclasses import FrozenInstanceError
from datetime import UTC
from uuid import UUID, uuid4

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


def create_agent_events():
    task_id = uuid4()
    step = AgentStep(
        instruction="Execute task",
        position=0,
    )
    plan = AgentPlan(
        task_id=task_id,
        steps=(step,),
    )
    success_result = AgentResult(
        task_id=task_id,
        success=True,
        output="Completed",
        completed_step_ids=(step.step_id,),
    )
    failure_result = AgentResult(
        task_id=task_id,
        success=False,
        error="Execution failed",
    )

    return [
        AgentTaskStarted(task_id=task_id),
        AgentPlanningStarted(task_id=task_id),
        AgentPlanCreated(task_id=task_id, plan=plan),
        AgentStepStarted(task_id=task_id, step=step),
        AgentStepCompleted(
            task_id=task_id,
            step=step,
            output="Step completed",
        ),
        AgentTaskCompleted(
            task_id=task_id,
            result=success_result,
        ),
        AgentTaskFailed(
            task_id=task_id,
            result=failure_result,
        ),
        AgentTaskCancelled(task_id=task_id),
    ]


def test_all_agent_events_inherit_base_event():
    events = create_agent_events()

    assert all(isinstance(event, Event) for event in events)


def test_all_agent_events_have_unique_ids():
    events = create_agent_events()
    event_ids = [event.event_id for event in events]

    assert all(isinstance(event_id, UUID) for event_id in event_ids)
    assert len(event_ids) == len(set(event_ids))


def test_all_agent_events_have_utc_timestamps():
    events = create_agent_events()

    for event in events:
        assert event.occurred_at.tzinfo is not None
        assert event.occurred_at.utcoffset() == UTC.utcoffset(
            event.occurred_at
        )


def test_event_metadata_is_immutable():
    event = AgentTaskStarted(task_id=uuid4())

    with pytest.raises(FrozenInstanceError):
        event.event_id = uuid4()

    with pytest.raises(FrozenInstanceError):
        event.occurred_at = event.occurred_at


def test_event_timestamp_is_timezone_aware():
    event = AgentTaskStarted(task_id=uuid4())

    assert event.occurred_at.tzinfo is not None
    assert event.occurred_at.utcoffset() is not None
