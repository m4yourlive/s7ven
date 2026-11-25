
"""Event models for the S7VEN autonomous agent."""

from dataclasses import dataclass
from uuid import UUID

from seven.core.agents.models import (
    AgentPlan,
    AgentResult,
    AgentStep,
)
from seven.core.events import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTaskStarted(Event):
    """Raised when an agent task begins."""

    task_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPlanningStarted(Event):
    """Raised when task planning begins."""

    task_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPlanCreated(Event):
    """Raised when a task plan has been created."""

    task_id: UUID
    plan: AgentPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentStepStarted(Event):
    """Raised when an execution step begins."""

    task_id: UUID
    step: AgentStep


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentStepCompleted(Event):
    """Raised when an execution step completes."""

    task_id: UUID
    step: AgentStep
    output: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTaskCompleted(Event):
    """Raised when an agent task completes successfully."""

    task_id: UUID
    result: AgentResult


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTaskFailed(Event):
    """Raised when an agent task fails."""

    task_id: UUID
    result: AgentResult


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTaskCancelled(Event):
    """Raised when an agent task is cancelled."""

    task_id: UUID
