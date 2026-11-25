
"""Recurring task lifecycle management."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from seven.core.integrations.scheduler.exceptions import (
    ScheduledTaskAlreadyExistsError,
    ScheduledTaskNotFoundError,
)
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
)
from seven.core.integrations.scheduler.timing import ScheduleTiming


@dataclass(frozen=True, slots=True)
class RecurringTaskProgress:
    """Track the execution progress of a recurring task."""

    schedule_id: UUID
    run_count: int = 0
    cancelled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, UUID):
            raise TypeError("schedule_id must be a UUID.")

        if isinstance(self.run_count, bool) or not isinstance(
            self.run_count, int
        ):
            raise TypeError("run_count must be an integer.")

        if self.run_count < 0:
            raise ValueError("run_count cannot be negative.")

        if not isinstance(self.cancelled, bool):
            raise TypeError("cancelled must be a boolean.")


class RecurringTaskManager:
    """Manage recurring task registration and execution progress."""

    def __init__(self) -> None:
        self._tasks: dict[UUID, ScheduledTask] = {}
        self._progress: dict[UUID, RecurringTaskProgress] = {}

    def register(self, task: ScheduledTask) -> RecurringTaskProgress:
        """Register a recurring task."""

        if not isinstance(task, ScheduledTask):
            raise TypeError("task must be a ScheduledTask.")

        if task.schedule_type is not ScheduleType.INTERVAL:
            raise ValueError("Only interval tasks can be registered.")

        if not task.enabled:
            raise ValueError("Disabled tasks cannot be registered.")

        schedule_id = task.schedule_id

        if schedule_id in self._tasks:
            raise ScheduledTaskAlreadyExistsError(
                f"Task '{schedule_id}' is already registered."
            )

        progress = RecurringTaskProgress(schedule_id=schedule_id)

        self._tasks[schedule_id] = task
        self._progress[schedule_id] = progress

        return progress

    def get_task(self, schedule_id: UUID) -> ScheduledTask:
        """Return a registered recurring task."""

        self._require_registered(schedule_id)
        return self._tasks[schedule_id]

    def get_progress(self, schedule_id: UUID) -> RecurringTaskProgress:
        """Return recurring task execution progress."""

        self._require_registered(schedule_id)
        return self._progress[schedule_id]

    def get_next_run(
        self,
        schedule_id: UUID,
        after: datetime,
    ) -> datetime | None:
        """Calculate the next scheduled execution."""

        self._require_registered(schedule_id)

        task = self._tasks[schedule_id]
        progress = self._progress[schedule_id]

        if progress.cancelled:
            return None

        return ScheduleTiming.get_next_run(
            task=task,
            after=after,
            run_count=progress.run_count,
        )

    def record_execution(
        self,
        schedule_id: UUID,
    ) -> RecurringTaskProgress:
        """Record one completed execution attempt."""

        self._require_registered(schedule_id)
        progress = self._progress[schedule_id]

        if progress.cancelled:
            raise ValueError("Cancelled tasks cannot record executions.")

        task = self._tasks[schedule_id]

        if (
            task.max_runs is not None
            and progress.run_count >= task.max_runs
        ):
            raise ValueError("Task has reached its maximum run count.")

        updated = RecurringTaskProgress(
            schedule_id=schedule_id,
            run_count=progress.run_count + 1,
        )

        self._progress[schedule_id] = updated
        return updated

    def cancel(self, schedule_id: UUID) -> bool:
        """Cancel a recurring task."""

        self._require_registered(schedule_id)
        progress = self._progress[schedule_id]

        if progress.cancelled:
            return False

        self._progress[schedule_id] = RecurringTaskProgress(
            schedule_id=schedule_id,
            run_count=progress.run_count,
            cancelled=True,
        )

        return True

    def unregister(self, schedule_id: UUID) -> bool:
        """Remove a recurring task and its progress."""

        self._require_registered(schedule_id)

        del self._tasks[schedule_id]
        del self._progress[schedule_id]

        return True

    def _require_registered(self, schedule_id: UUID) -> None:
        if not isinstance(schedule_id, UUID):
            raise TypeError("schedule_id must be a UUID.")

        if schedule_id not in self._tasks:
            raise ScheduledTaskNotFoundError(
                f"Task '{schedule_id}' is not registered."
            )
