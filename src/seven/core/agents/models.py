"""Domain models for the S7VEN autonomous agent."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

from seven.core.ai.models import AIMessage


def _validate_timestamp(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime instance.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _freeze_metadata(
    metadata: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")

    if not all(isinstance(key, str) for key in metadata):
        raise TypeError(f"{field_name} keys must be strings.")

    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTask:
    """Represent a user goal submitted to the autonomous agent."""

    goal: str
    task_id: UUID = field(default_factory=uuid4)
    context: tuple[AIMessage, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must not be empty.")

        if not isinstance(self.context, tuple):
            raise TypeError("context must be a tuple.")

        if not all(
            isinstance(message, AIMessage)
            for message in self.context
        ):
            raise TypeError("context must contain AIMessage instances.")

        _validate_timestamp(self.created_at, "created_at")

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentStep:
    """Represent one executable step in an agent plan."""

    instruction: str
    position: int
    step_id: UUID = field(default_factory=uuid4)
    dependencies: tuple[UUID, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, UUID):
            raise TypeError("step_id must be a UUID.")

        if (
            not isinstance(self.instruction, str)
            or not self.instruction.strip()
        ):
            raise ValueError("instruction must not be empty.")

        if (
            not isinstance(self.position, int)
            or isinstance(self.position, bool)
        ):
            raise TypeError("position must be an integer.")

        if self.position < 0:
            raise ValueError("position cannot be negative.")

        if not isinstance(self.dependencies, tuple):
            raise TypeError("dependencies must be a tuple.")

        if not all(
            isinstance(dependency, UUID)
            for dependency in self.dependencies
        ):
            raise TypeError("dependencies must contain UUID values.")

        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must not contain duplicates.")

        if self.step_id in self.dependencies:
            raise ValueError("A step cannot depend on itself.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPlan:
    """Represent an ordered execution plan for an agent task."""

    task_id: UUID
    steps: tuple[AgentStep, ...]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple.")

        if not self.steps:
            raise ValueError("An agent plan must contain at least one step.")

        if not all(isinstance(step, AgentStep) for step in self.steps):
            raise TypeError("steps must contain AgentStep instances.")

        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("An agent plan cannot contain duplicate steps.")

        positions = [step.position for step in self.steps]
        if len(set(positions)) != len(positions):
            raise ValueError("Step positions must be unique.")

        steps_by_id = {
            step.step_id: step
            for step in self.steps
        }

        for step in self.steps:
            for dependency_id in step.dependencies:
                dependency = steps_by_id.get(dependency_id)

                if dependency is None:
                    raise ValueError(
                        "Every step dependency must exist in the plan."
                    )

                if dependency.position >= step.position:
                    raise ValueError(
                        "A step can only depend on an earlier step."
                    )

        _validate_timestamp(self.created_at, "created_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentResult:
    """Represent the final outcome of an agent execution."""

    task_id: UUID
    success: bool
    output: str = ""
    completed_step_ids: tuple[UUID, ...] = ()
    failed_step_id: UUID | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean.")

        if not isinstance(self.output, str):
            raise TypeError("output must be a string.")

        if not isinstance(self.completed_step_ids, tuple):
            raise TypeError("completed_step_ids must be a tuple.")

        if not all(
            isinstance(step_id, UUID)
            for step_id in self.completed_step_ids
        ):
            raise TypeError(
                "completed_step_ids must contain UUID values."
            )

        if len(set(self.completed_step_ids)) != len(
            self.completed_step_ids
        ):
            raise ValueError(
                "completed_step_ids must not contain duplicates."
            )

        if self.failed_step_id is not None and not isinstance(
            self.failed_step_id,
            UUID,
        ):
            raise TypeError("failed_step_id must be a UUID or None.")

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")

        if self.success and self.error:
            raise ValueError("A successful result cannot contain an error.")

        if not self.success and not self.error:
            raise ValueError("A failed result must contain an error.")

        _validate_timestamp(self.completed_at, "completed_at")

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "metadata"),
        )