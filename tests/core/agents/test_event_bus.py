
"""Tests for autonomous agent event publishing."""

from uuid import uuid4

import pytest

from seven.core.agents.events import (
    AgentPlanningStarted,
    AgentTaskStarted,
)
from seven.core.events import Event, EventBus


def test_agent_events_can_be_published():
    event_bus = EventBus()
    received_events: list[Event] = []

    event_bus.subscribe(
        AgentTaskStarted,
        received_events.append,
    )

    event = AgentTaskStarted(task_id=uuid4())
    published_count = event_bus.publish(event)

    assert published_count == 1
    assert received_events == [event]


def test_event_bus_filters_event_types():
    event_bus = EventBus()
    received_events: list[Event] = []

    event_bus.subscribe(
        AgentTaskStarted,
        received_events.append,
    )

    event_bus.publish(
        AgentPlanningStarted(task_id=uuid4())
    )

    assert received_events == []


def test_multiple_agent_event_types_can_be_published():
    event_bus = EventBus()
    received_events: list[Event] = []

    event_bus.subscribe(
        AgentTaskStarted,
        received_events.append,
    )
    event_bus.subscribe(
        AgentPlanningStarted,
        received_events.append,
    )

    started_event = AgentTaskStarted(task_id=uuid4())
    planning_event = AgentPlanningStarted(task_id=uuid4())

    event_bus.publish(started_event)
    event_bus.publish(planning_event)

    assert received_events == [
        started_event,
        planning_event,
    ]


def test_event_bus_handles_multiple_subscribers():
    event_bus = EventBus()
    first_subscriber: list[Event] = []
    second_subscriber: list[Event] = []

    event_bus.subscribe(
        AgentTaskStarted,
        first_subscriber.append,
    )
    event_bus.subscribe(
        AgentTaskStarted,
        second_subscriber.append,
    )

    event = AgentTaskStarted(task_id=uuid4())
    published_count = event_bus.publish(event)

    assert published_count == 2
    assert first_subscriber == [event]
    assert second_subscriber == [event]


def test_event_bus_does_not_publish_unrelated_events():
    event_bus = EventBus()
    received_events: list[Event] = []

    event_bus.subscribe(
        AgentTaskStarted,
        received_events.append,
    )

    event_bus.publish(
        AgentPlanningStarted(task_id=uuid4())
    )

    assert not received_events
