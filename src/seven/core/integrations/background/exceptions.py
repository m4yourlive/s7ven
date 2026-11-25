
"""Background task exceptions."""


class BackgroundTaskError(Exception):
    """Base exception for background task operations."""


class BackgroundTaskNotFoundError(BackgroundTaskError):
    """Raised when a background task cannot be found."""


class BackgroundTaskAlreadyExistsError(BackgroundTaskError):
    """Raised when a task is already registered."""


class BackgroundTaskExecutionError(BackgroundTaskError):
    """Raised when background task execution fails."""


class BackgroundTaskQueueFullError(BackgroundTaskError):
    """Raised when the task queue reaches its capacity."""


class BackgroundTaskCancellationError(BackgroundTaskError):
    """Raised when a task cannot be cancelled."""


class BackgroundTaskManagerError(BackgroundTaskError):
    """Raised when background manager operations fail."""
