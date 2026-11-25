
"""Domain models for scheduled task automation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from typing import Any


class ScheduleType(str, Enum):
    """Supported scheduling strategies."""

    ONCE = "once"
    INTERVAL = "interval"


class ScheduledTaskStatus(str, Enum):
    """Represent scheduled task lifecycle states."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    """Represent a task configured for scheduled execution."""

    name: str
    action: str
    schedule_type: ScheduleType
    run_at: datetime
    parameters: dict[str, Any] = field(default_factory=dict)
    interval_seconds: float | None = None
    max_runs: int | None = None
    enabled: bool = True
    schedule_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Task name must be a string.")

        if not self.name.strip():
            raise ValueError("Task name cannot be empty.")

        if not isinstance(self.action, str):
            raise TypeError("Task action must be a string.")

        if not self.action.strip():
            raise ValueError("Task action cannot be empty.")

        if not isinstance(self.schedule_type, ScheduleType):
            raise TypeError("Schedule type must be a ScheduleType.")

        if not isinstance(self.run_at, datetime):
            raise TypeError("run_at must be a datetime.")

        if self.run_at.tzinfo is None or self.run_at.utcoffset() is None:
            raise ValueError("run_at must be timezone-aware.")

        if not isinstance(self.parameters, dict):
            raise TypeError("Task parameters must be a dictionary.")

        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")

        if not isinstance(self.schedule_id, UUID):
            raise TypeError("schedule_id must be a UUID.")

        if self.interval_seconds is not None:
            if (
                isinstance(self.interval_seconds, bool)
                or not isinstance(self.interval_seconds, (int, float))
            ):
                raise TypeError("interval_seconds must be a number.")

            if self.interval_seconds <= 0:
                raise ValueError("interval_seconds must be positive.")

        if self.max_runs is not None:
            if isinstance(self.max_runs, bool) or not isinstance(
                self.max_runs, int
            ):
                raise TypeError("max_runs must be an integer.")

            if self.max_runs <= 0:
                raise ValueError("max_runs must be positive.")

        if self.schedule_type is ScheduleType.ONCE:
            if self.interval_seconds is not None:
                raise ValueError(
                    "One-time schedules cannot define an interval."
                )

            if self.max_runs not in (None, 1):
                raise ValueError(
                    "One-time schedules can execute only once."
                )

        elif self.schedule_type is ScheduleType.INTERVAL:
            if self.interval_seconds is None:
                raise ValueError(
                    "Interval schedules require interval_seconds."
                )

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "action", self.action.strip())
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True, slots=True)
class ScheduledExecution:
    """Represent the result of one scheduled task execution."""

    schedule_id: UUID
    scheduled_for: datetime
    status: ScheduledTaskStatus
    execution_id: UUID = field(default_factory=uuid4)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    output: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, UUID):
            raise TypeError("schedule_id must be a UUID.")

        if not isinstance(self.execution_id, UUID):
            raise TypeError("execution_id must be a UUID.")

        if not isinstance(self.scheduled_for, datetime):
            raise TypeError("scheduled_for must be a datetime.")

        if (
            self.scheduled_for.tzinfo is None
            or self.scheduled_for.utcoffset() is None
        ):
            raise ValueError("scheduled_for must be timezone-aware.")

        if not isinstance(self.status, ScheduledTaskStatus):
            raise TypeError("status must be a ScheduledTaskStatus.")

        for field_name, value in (
            ("started_at", self.started_at),
            ("finished_at", self.finished_at),
        ):
            if value is not None:
                if not isinstance(value, datetime):
                    raise TypeError(f"{field_name} must be a datetime.")

                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(
                        f"{field_name} must be timezone-aware."
                    )

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")
