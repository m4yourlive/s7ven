
"""Priority-based background task queue."""

import asyncio
import heapq
from uuid import UUID

from seven.core.integrations.background.exceptions import (
    BackgroundTaskAlreadyExistsError,
    BackgroundTaskNotFoundError,
    BackgroundTaskQueueFullError,
)
from seven.core.integrations.background.models import BackgroundTask


class BackgroundTaskQueue:
    """Manage pending background tasks using priority ordering."""

    def __init__(self, max_size: int = 1000) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int):
            raise TypeError("max_size must be an integer.")

        if max_size <= 0:
            raise ValueError("max_size must be positive.")

        self._max_size = max_size
        self._queue: list[tuple[int, int, UUID, BackgroundTask]] = []
        self._tasks: dict[UUID, BackgroundTask] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Return the number of pending tasks."""

        return len(self._tasks)

    @property
    def is_empty(self) -> bool:
        """Return whether the queue is empty."""

        return not self._tasks

    @property
    def is_full(self) -> bool:
        """Return whether the queue has reached capacity."""

        return self.size >= self._max_size

    async def submit(self, task: BackgroundTask) -> UUID:
        """Add a background task to the queue."""

        if not isinstance(task, BackgroundTask):
            raise TypeError("task must be a BackgroundTask.")

        async with self._lock:
            if task.task_id in self._tasks:
                raise BackgroundTaskAlreadyExistsError(
                    f"Task '{task.task_id}' is already queued."
                )

            if self.is_full:
                raise BackgroundTaskQueueFullError(
                    "Background task queue is full."
                )

            entry = (
                -task.priority,
                self._sequence,
                task.task_id,
                task,
            )

            self._sequence += 1
            self._tasks[task.task_id] = task
            heapq.heappush(self._queue, entry)

            return task.task_id

    async def get(self) -> BackgroundTask:
        """Retrieve the highest-priority pending task."""

        async with self._lock:
            while self._queue:
                _, _, task_id, task = heapq.heappop(self._queue)

                if task_id in self._tasks:
                    del self._tasks[task_id]
                    return task

            raise BackgroundTaskNotFoundError(
                "No pending background tasks are available."
            )

    async def remove(self, task_id: UUID) -> bool:
        """Remove a pending task from the queue."""

        self._validate_id(task_id)

        async with self._lock:
            if task_id not in self._tasks:
                return False

            del self._tasks[task_id]
            return True

    async def contains(self, task_id: UUID) -> bool:
        """Check whether a task is pending."""

        self._validate_id(task_id)

        async with self._lock:
            return task_id in self._tasks

    async def snapshot(self) -> tuple[BackgroundTask, ...]:
        """Return pending tasks in execution order."""

        async with self._lock:
            entries = sorted(self._queue)

            return tuple(
                task
                for _, _, task_id, task in entries
                if task_id in self._tasks
            )

    async def clear(self) -> tuple[BackgroundTask, ...]:
        """Remove and return all pending tasks."""

        async with self._lock:
            tasks = tuple(self._tasks.values())

            self._tasks.clear()
            self._queue.clear()

            return tasks

    @staticmethod
    def _validate_id(task_id: UUID) -> None:
        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID.")
