
"""Background task management."""

from seven.core.integrations.background.cancellation import (
    BackgroundTaskCancellationManager,
)
from seven.core.integrations.background.exceptions import (
    BackgroundTaskAlreadyExistsError,
    BackgroundTaskCancellationError,
    BackgroundTaskError,
    BackgroundTaskExecutionError,
    BackgroundTaskManagerError,
    BackgroundTaskNotFoundError,
    BackgroundTaskQueueFullError,
)
from seven.core.integrations.background.executor import (
    BackgroundTaskEngine,
)
from seven.core.integrations.background.interfaces import (
    BackgroundTaskExecutor,
    BackgroundTaskStore,
)
from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskResult,
    BackgroundTaskStatus,
)
from seven.core.integrations.background.progress import (
    BackgroundTaskProgressTracker,
    TaskProgress,
)
from seven.core.integrations.background.queue import (
    BackgroundTaskQueue,
)
from seven.core.integrations.background.worker import (
    BackgroundWorkerPool,
)

__all__ = [
    "BackgroundTask",
    "BackgroundTaskAlreadyExistsError",
    "BackgroundTaskCancellationError",
    "BackgroundTaskCancellationManager",
    "BackgroundTaskEngine",
    "BackgroundTaskError",
    "BackgroundTaskExecutionError",
    "BackgroundTaskExecutor",
    "BackgroundTaskManagerError",
    "BackgroundTaskNotFoundError",
    "BackgroundTaskProgressTracker",
    "BackgroundTaskQueue",
    "BackgroundTaskQueueFullError",
    "BackgroundTaskResult",
    "BackgroundTaskStatus",
    "BackgroundTaskStore",
    "BackgroundWorkerPool",
    "TaskProgress",
]
