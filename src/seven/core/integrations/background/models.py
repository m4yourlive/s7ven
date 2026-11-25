
"""Background task domain models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class BackgroundTaskStatus(str, Enum):
    """Represent background task lifecycle states."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class BackgroundTask:
    """Represent a unit of background work."""

    name: str
    action: str
    parameters: dict = field(default_factory=dict)
    task_id: UUID = field(default_factory=uuid4)
    priority: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Task name must be a string.")

        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Task name cannot be empty.")

        object.__setattr__(self, "name", normalized_name)

        if not isinstance(self.action, str):
            raise TypeError("Task action must be a string.")

        normalized_action = self.action.strip()

        if not normalized_action:
            raise ValueError("Task action cannot be empty.")

        object.__setattr__(self, "action", normalized_action)

        if not isinstance(self.parameters, dict):
            raise TypeError("Task parameters must be a dictionary.")

        if not isinstance(self.task_id, UUID):
            raise TypeError("Task ID must be a UUID.")

        if isinstance(self.priority, bool) or not isinstance(
            self.priority, int
        ):
            raise TypeError("Task priority must be an integer.")

        if not isinstance(self.created_at, datetime):
            raise TypeError("Task creation time must be a datetime.")

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("Task creation time must be timezone-aware.")

        object.__setattr__(
            self,
            "parameters",
            dict(self.parameters),
        )


@dataclass(frozen=True, slots=True)
class BackgroundTaskResult:
    """Represent the result of a background task."""

    task_id: UUID
    status: BackgroundTaskStatus
    output: object = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("Task ID must be a UUID.")

        if not isinstance(self.status, BackgroundTaskStatus):
            raise TypeError("Task status must be a BackgroundTaskStatus.")

        for field_name in ("started_at", "finished_at"):
            value = getattr(self, field_name)

            if value is not None:
                if not isinstance(value, datetime):
                    raise TypeError(
                        f"{field_name} must be a datetime or None."
                    )

                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(
                        f"{field_name} must be timezone-aware."
                    )

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("Task error must be a string or None.")

        if self.error is not None and not self.error.strip():
            raise ValueError("Task error cannot be empty.")
