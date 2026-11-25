"""State management for autonomous agent execution."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from seven.core.agents.models import AgentPlan, AgentResult


class AgentStatus(StrEnum):
    """Available autonomous agent lifecycle states."""

    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.PENDING: frozenset({
        AgentStatus.PLANNING,
        AgentStatus.CANCELLED,
    }),
    AgentStatus.PLANNING: frozenset({
        AgentStatus.READY,
        AgentStatus.FAILED,
        AgentStatus.CANCELLED,
    }),
    AgentStatus.READY: frozenset({
        AgentStatus.RUNNING,
        AgentStatus.CANCELLED,
    }),
    AgentStatus.RUNNING: frozenset({
        AgentStatus.COMPLETED,
        AgentStatus.FAILED,
        AgentStatus.CANCELLED,
    }),
    AgentStatus.COMPLETED: frozenset(),
    AgentStatus.FAILED: frozenset(),
    AgentStatus.CANCELLED: frozenset(),
}


@dataclass(slots=True)
class AgentState:
    """Track the lifecycle and execution progress of an agent task."""

    task_id: UUID
    status: AgentStatus = AgentStatus.PENDING
    plan: AgentPlan | None = None
    result: AgentResult | None = None
    current_step_id: UUID | None = None
    completed_step_ids: tuple[UUID, ...] = ()
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        if not isinstance(self.status, AgentStatus):
            raise TypeError("status must be an AgentStatus.")

        if self.plan is not None:
            if not isinstance(self.plan, AgentPlan):
                raise TypeError("plan must be an AgentPlan or None.")

            if self.plan.task_id != self.task_id:
                raise ValueError("Plan task_id does not match state task_id.")

        if self.result is not None:
            if not isinstance(self.result, AgentResult):
                raise TypeError("result must be an AgentResult or None.")

            if self.result.task_id != self.task_id:
                raise ValueError(
                    "Result task_id does not match state task_id."
                )

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

        if self.current_step_id is not None and not isinstance(
            self.current_step_id,
            UUID,
        ):
            raise TypeError("current_step_id must be a UUID or None.")

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")

        self._validate_optional_timestamp(self.started_at, "started_at")
        self._validate_optional_timestamp(
            self.completed_at,
            "completed_at",
        )
        self._validate_optional_timestamp(self.updated_at, "updated_at")

    @staticmethod
    def _validate_optional_timestamp(
        value: datetime | None,
        field_name: str,
    ) -> None:
        if value is None:
            return

        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime or None.")

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

    @property
    def is_terminal(self) -> bool:
        """Return whether the task has reached a terminal state."""
        return self.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }

    @property
    def is_active(self) -> bool:
        """Return whether the task is currently being processed."""
        return self.status in {
            AgentStatus.PLANNING,
            AgentStatus.READY,
            AgentStatus.RUNNING,
        }

    def transition(self, target: AgentStatus) -> None:
        """Move the task into a valid lifecycle state."""
        if not isinstance(target, AgentStatus):
            raise TypeError("target must be an AgentStatus.")

        allowed = _ALLOWED_TRANSITIONS[self.status]

        if target not in allowed:
            raise ValueError(
                f"Invalid agent state transition: "
                f"{self.status.value} -> {target.value}."
            )

        now = datetime.now(UTC)

        self.status = target
        self.updated_at = now

        if target == AgentStatus.RUNNING and self.started_at is None:
            self.started_at = now

        if target in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }:
            self.completed_at = now

    def set_plan(self, plan: AgentPlan) -> None:
        """Attach a validated plan and mark the task as ready."""
        if self.status != AgentStatus.PLANNING:
            raise ValueError(
                "A plan can only be assigned while planning."
            )

        if not isinstance(plan, AgentPlan):
            raise TypeError("plan must be an AgentPlan instance.")

        if plan.task_id != self.task_id:
            raise ValueError("Plan task_id does not match state task_id.")

        self.plan = plan
        self.transition(AgentStatus.READY)

    def start_step(self, step_id: UUID) -> None:
        """Mark a planned step as the currently executing step."""
        if self.status != AgentStatus.RUNNING:
            raise ValueError(
                "Steps can only start while the agent is running."
            )

        if self.plan is None:
            raise ValueError("Cannot start a step without an agent plan.")

        if self.current_step_id is not None:
            raise ValueError("Another step is already running.")

        if not isinstance(step_id, UUID):
            raise TypeError("step_id must be a UUID.")

        step = next(
            (
                item
                for item in self.plan.steps
                if item.step_id == step_id
            ),
            None,
        )

        if step is None:
            raise ValueError("Step does not belong to the current plan.")

        if step_id in self.completed_step_ids:
            raise ValueError("Step has already been completed.")

        missing_dependencies = set(step.dependencies) - set(
            self.completed_step_ids
        )

        if missing_dependencies:
            raise ValueError(
                "Step dependencies have not been completed."
            )

        self.current_step_id = step_id
        self.updated_at = datetime.now(UTC)

    def complete_step(self, step_id: UUID) -> None:
        """Mark the currently executing step as completed."""
        if self.status != AgentStatus.RUNNING:
            raise ValueError(
                "Steps can only complete while the agent is running."
            )

        if not isinstance(step_id, UUID):
            raise TypeError("step_id must be a UUID.")

        if self.current_step_id != step_id:
            raise ValueError(
                "The completed step must match the current step."
            )

        self.completed_step_ids += (step_id,)
        self.current_step_id = None
        self.updated_at = datetime.now(UTC)

    def finish(self, result: AgentResult) -> None:
        """Store the final result and transition to a terminal state."""
        if self.status != AgentStatus.RUNNING:
            raise ValueError(
                "An agent task can only finish while running."
            )

        if not isinstance(result, AgentResult):
            raise TypeError("result must be an AgentResult instance.")

        if result.task_id != self.task_id:
            raise ValueError(
                "Result task_id does not match state task_id."
            )

        if result.completed_step_ids != self.completed_step_ids:
            raise ValueError(
                "Result completed steps do not match agent state."
            )

        if result.success:
            if self.current_step_id is not None:
                raise ValueError(
                    "Cannot complete while a step is still running."
                )

            if self.plan is None:
                raise ValueError(
                    "Cannot complete without an execution plan."
                )

            planned_step_ids = {
                step.step_id for step in self.plan.steps
            }

            if set(self.completed_step_ids) != planned_step_ids:
                raise ValueError(
                    "Cannot complete before all planned steps finish."
                )

            target = AgentStatus.COMPLETED
        else:
            target = AgentStatus.FAILED

        self.result = result
        self.error = result.error
        self.current_step_id = None
        self.transition(target)

    def cancel(self) -> None:
        """Cancel a task that has not reached a terminal state."""
        self.transition(AgentStatus.CANCELLED)