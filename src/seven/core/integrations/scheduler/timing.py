
"""Scheduling time calculations."""

from datetime import datetime, timedelta

from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
)


class ScheduleTiming:
    """Calculate scheduled task execution times."""

    @staticmethod
    def get_next_run(
        task: ScheduledTask,
        after: datetime,
        run_count: int = 0,
    ) -> datetime | None:
        """Return the next execution time after the supplied timestamp."""

        if not isinstance(task, ScheduledTask):
            raise TypeError("task must be a ScheduledTask.")

        if not isinstance(after, datetime):
            raise TypeError("after must be a datetime.")

        if after.tzinfo is None or after.utcoffset() is None:
            raise ValueError("after must be timezone-aware.")

        if after.utcoffset() != task.run_at.utcoffset():
            after = after.astimezone(task.run_at.tzinfo)

        if isinstance(run_count, bool) or not isinstance(run_count, int):
            raise TypeError("run_count must be an integer.")

        if run_count < 0:
            raise ValueError("run_count cannot be negative.")

        if task.max_runs is not None and run_count >= task.max_runs:
            return None

        if task.schedule_type is ScheduleType.ONCE:
            if run_count > 0 or task.run_at <= after:
                return None

            return task.run_at

        if task.schedule_type is ScheduleType.INTERVAL:
            interval = timedelta(seconds=task.interval_seconds)

            if after < task.run_at:
                return task.run_at

            elapsed = after - task.run_at
            intervals_elapsed = (
                elapsed // interval
            ) + 1

            next_run = task.run_at + (
                interval * intervals_elapsed
            )

            return next_run

        return None
