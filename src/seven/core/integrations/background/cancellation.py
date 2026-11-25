
"""Background task cancellation management."""

from uuid import UUID

from seven.core.integrations.background.models import (
    BackgroundTaskStatus,
)
from seven.core.integrations.background.progress import (
    BackgroundTaskProgressTracker,
)
from seven.core.integrations.background.worker import (
    BackgroundWorkerPool,
)


class BackgroundTaskCancellationManager:
    """Coordinate task cancellation and worker shutdown."""

    def __init__(
        self,
        worker_pool: BackgroundWorkerPool,
        progress_tracker: BackgroundTaskProgressTracker | None = None,
    ) -> None:
        if not isinstance(worker_pool, BackgroundWorkerPool):
            raise TypeError("worker_pool must be a BackgroundWorkerPool.")

        if (
            progress_tracker is not None
            and not isinstance(
                progress_tracker,
                BackgroundTaskProgressTracker,
            )
        ):
            raise TypeError(
                "progress_tracker must be a BackgroundTaskProgressTracker."
            )

        self._worker_pool = worker_pool
        self._progress_tracker = progress_tracker

    async def cancel(self, task_id: UUID) -> bool:
        """Cancel a task and update its tracked status."""

        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        cancelled = await self._worker_pool.cancel_task(task_id)

        if cancelled and self._progress_tracker is not None:
            progress = await self._progress_tracker.get(task_id)

            if progress is not None:
                await self._progress_tracker.update_status(
                    task_id,
                    BackgroundTaskStatus.CANCELLED,
                    "Task cancelled.",
                )

        return cancelled

    async def shutdown(self, cancel_running: bool = False) -> None:
        """Shut down the worker pool."""

        await self._worker_pool.stop(cancel_running=cancel_running)
