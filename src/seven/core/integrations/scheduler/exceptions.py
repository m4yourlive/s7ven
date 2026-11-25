
"""Exceptions for scheduled task automation."""


class SchedulerError(Exception):
    """Base exception for scheduler operations."""


class ScheduleValidationError(SchedulerError, ValueError):
    """Raised when a scheduled task definition is invalid."""


class ScheduledTaskNotFoundError(SchedulerError):
    """Raised when a scheduled task cannot be found."""


class ScheduledTaskAlreadyExistsError(SchedulerError):
    """Raised when a scheduled task already exists."""


class ScheduledTaskExecutionError(SchedulerError):
    """Raised when a scheduled task execution fails."""
