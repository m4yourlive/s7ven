
"""Background task progress tracking."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from seven.core.integrations.background.models import (
    BackgroundTaskStatus,
)


@dataclass(frozen=True, slots=True)
class TaskProgress:
    """Represent the current progress of a background task."""

    task_id: UUID
    status: BackgroundTaskStatus
    percentage: float = 0.0
    message: str | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID.")
        if not isinstance(self.status, BackgroundTaskStatus):
            raise TypeError("status must be a BackgroundTaskStatus.")
        if isinstance(self.percentage, bool) or not isinstance(
            self.percentage, (int, float)
        ):
            raise TypeError("percentage must be numeric.")
        if not 0 <= self.percentage <= 100:
            raise ValueError("percentage must be between 0 and 100.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None.")
        if self.updated_at is not None:
            if not isinstance(self.updated_at, datetime):
                raise TypeError("updated_at must be a datetime or None.")
            if (
                self.updated_at.tzinfo is None
                or self.updated_at.utcoffset() is None
            ):
                raise ValueError("updated_at must be timezone-aware.")


class BackgroundTaskProgressTracker:
    """Track status and progress for background tasks."""

    def __init__(self) -> None:
        self._progress: dict[UUID, TaskProgress] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        task_id: UUID,
        status: BackgroundTaskStatus = BackgroundTaskStatus.PENDING,
    ) -> TaskProgress:
        """Register a task for progress tracking."""

        self._validate_task_id(task_id)

        if not isinstance(status, BackgroundTaskStatus):
            raise TypeError("status must be a BackgroundTaskStatus.")

        async with self._lock:
            if task_id in self._progress:
                raise ValueError("Task is already registered.")

            progress = TaskProgress(
                task_id=task_id,
                status=status,
                updated_at=self._now(),
            )
            self._progress[task_id] = progress
            return progress

    async def update_progress(
        self,
        task_id: UUID,
        percentage: float,
        message: str | None = None,
    ) -> TaskProgress:
        """Update a task's progress percentage and message."""

        self._validate_task_id(task_id)

        if isinstance(percentage, bool) or not isinstance(
            percentage, (int, float)
        ):
            raise TypeError("percentage must be numeric.")
        if not 0 <= percentage <= 100:
            raise ValueError("percentage must be between 0 and 100.")
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be a string or None.")

        async with self._lock:
            current = self._require_task(task_id)

            if current.status in {
                BackgroundTaskStatus.COMPLETED,
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.CANCELLED,
            }:
                raise ValueError("Cannot update a finished task.")

            updated = TaskProgress(
                task_id=task_id,
                status=current.status,
                percentage=float(percentage),
                message=message,
                updated_at=self._now(),
            )
            self._progress[task_id] = updated
            return updated

    async def update_status(
        self,
        task_id: UUID,
        status: BackgroundTaskStatus,
        message: str | None = None,
    ) -> TaskProgress:
        """Update a task's execution status."""

        self._validate_task_id(task_id)

        if not isinstance(status, BackgroundTaskStatus):
            raise TypeError("status must be a BackgroundTaskStatus.")
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be a string or None.")

        async with self._lock:
            current = self._require_task(task_id)

            if current.status in {
                BackgroundTaskStatus.COMPLETED,
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.CANCELLED,
            }:
                raise ValueError("Cannot change the status of a finished task.")

            percentage = current.percentage

            if status is BackgroundTaskStatus.COMPLETED:
                percentage = 100.0

            updated = TaskProgress(
                task_id=task_id,
                status=status,
                percentage=percentage,
                message=message,
                updated_at=self._now(),
            )
            self._progress[task_id] = updated
            return updated

    async def get(self, task_id: UUID) -> TaskProgress | None:
        """Return progress for a task, if registered."""

        self._validate_task_id(task_id)

        async with self._lock:
            return self._progress.get(task_id)

    async def list_all(self) -> tuple[TaskProgress, ...]:
        """Return a snapshot of all tracked tasks."""

        async with self._lock:
            return tuple(self._progress.values())

    async def remove(self, task_id: UUID) -> bool:
        """Remove a task from progress tracking."""

        self._validate_task_id(task_id)

        async with self._lock:
            return self._progress.pop(task_id, None) is not None

    async def clear(self) -> tuple[TaskProgress, ...]:
        """Remove and return all tracked progress records."""

        async with self._lock:
            records = tuple(self._progress.values())
            self._progress.clear()
            return records

    def _require_task(self, task_id: UUID) -> TaskProgress:
        progress = self._progress.get(task_id)

        if progress is None:
            raise KeyError(f"Task {task_id} is not registered.")

        return progress

    @staticmethod
    def _validate_task_id(task_id: UUID) -> None:
        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID.")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
