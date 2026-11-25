
"""Validation for scheduled task definitions."""

from seven.core.integrations.scheduler.exceptions import (
    ScheduleValidationError,
)
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
)


class DefaultScheduleValidator:
    """Validate scheduled task definitions."""

    def validate(self, task: ScheduledTask) -> None:
        """Validate a scheduled task before registration."""

        if not isinstance(task, ScheduledTask):
            raise ScheduleValidationError(
                "Task must be a ScheduledTask instance."
            )

        if not task.name.strip():
            raise ScheduleValidationError(
                "Scheduled task name cannot be empty."
            )

        if not task.action.strip():
            raise ScheduleValidationError(
                "Scheduled task action cannot be empty."
            )

        if not isinstance(task.schedule_type, ScheduleType):
            raise ScheduleValidationError(
                "Unsupported schedule type."
            )

        if (
            not isinstance(task.run_at, __import__("datetime").datetime)
            or task.run_at.tzinfo is None
            or task.run_at.utcoffset() is None
        ):
            raise ScheduleValidationError(
                "Scheduled execution time must be timezone-aware."
            )

        if not isinstance(task.parameters, dict):
            raise ScheduleValidationError(
                "Task parameters must be a dictionary."
            )

        if task.max_runs is not None:
            if isinstance(task.max_runs, bool) or not isinstance(
                task.max_runs, int
            ):
                raise ScheduleValidationError(
                    "Maximum runs must be an integer."
                )

            if task.max_runs <= 0:
                raise ScheduleValidationError(
                    "Maximum runs must be greater than zero."
                )

        if task.schedule_type is ScheduleType.ONCE:
            if task.interval_seconds is not None:
                raise ScheduleValidationError(
                    "One-time schedules cannot define an interval."
                )

            if task.max_runs not in (None, 1):
                raise ScheduleValidationError(
                    "One-time schedules can execute only once."
                )

        elif task.schedule_type is ScheduleType.INTERVAL:
            if task.interval_seconds is None:
                raise ScheduleValidationError(
                    "Interval schedules require an interval."
                )

            if (
                isinstance(task.interval_seconds, bool)
                or not isinstance(task.interval_seconds, (int, float))
            ):
                raise ScheduleValidationError(
                    "Interval must be a number."
                )

            if task.interval_seconds <= 0:
                raise ScheduleValidationError(
                    "Interval must be greater than zero."
                )
