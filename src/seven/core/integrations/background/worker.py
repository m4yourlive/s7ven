
"""Concurrent background task worker pool."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from uuid import UUID

from seven.core.integrations.background.exceptions import (
    BackgroundTaskError,
    BackgroundTaskNotFoundError,
)
from seven.core.integrations.background.executor import (
    BackgroundTaskEngine,
)
from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskResult,
    BackgroundTaskStatus,
)
from seven.core.integrations.background.queue import (
    BackgroundTaskQueue,
)


ResultHandler = Callable[[BackgroundTaskResult], Awaitable[None] | None]


class BackgroundWorkerPool:
    """Run queued background tasks using a bounded worker pool."""

    def __init__(
        self,
        task_queue: BackgroundTaskQueue,
        executor: BackgroundTaskEngine,
        worker_count: int = 3,
        result_handler: ResultHandler | None = None,
    ) -> None:
        if not isinstance(task_queue, BackgroundTaskQueue):
            raise TypeError("task_queue must be a BackgroundTaskQueue.")
        if not isinstance(executor, BackgroundTaskEngine):
            raise TypeError("executor must be a BackgroundTaskEngine.")
        if isinstance(worker_count, bool) or not isinstance(worker_count, int):
            raise TypeError("worker_count must be an integer.")
        if worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        if result_handler is not None and not callable(result_handler):
            raise TypeError("result_handler must be callable.")

        self._queue = task_queue
        self._executor = executor
        self._worker_count = worker_count
        self._result_handler = result_handler
        self._workers: list[asyncio.Task[None]] = []
        self._running_tasks: dict[UUID, asyncio.Task[None]] = {}
        self._started = False
        self._stopping = False
        self._state_lock = asyncio.Lock()

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopping

    @property
    def active_worker_count(self) -> int:
        return sum(not worker.done() for worker in self._workers)

    @property
    def active_task_count(self) -> int:
        return sum(not task.done() for task in self._running_tasks.values())

    async def start(self) -> None:
        """Start all workers."""

        async with self._state_lock:
            if self._started:
                return

            self._stopping = False
            self._workers = [
                asyncio.create_task(
                    self._worker_loop(),
                    name=f"seven-background-worker-{index + 1}",
                )
                for index in range(self._worker_count)
            ]
            self._started = True

    async def submit(self, task: BackgroundTask) -> UUID:
        """Submit a task to the shared queue."""

        if self._stopping:
            raise BackgroundTaskError("Worker pool is stopping.")

        return await self._queue.submit(task)

    async def cancel_task(self, task_id: UUID) -> bool:
        """Cancel a queued or currently running task."""

        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        running = self._running_tasks.get(task_id)

        if running is not None and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            return True

        queued_tasks = await self._queue.snapshot()
        queued_task = next(
            (task for task in queued_tasks if task.task_id == task_id),
            None,
        )

        if queued_task is None:
            return False

        removed = await self._queue.remove(task_id)

        if not removed:
            return False

        now = datetime.now(timezone.utc)
        await self._notify_result(
            BackgroundTaskResult(
                task_id=task_id,
                status=BackgroundTaskStatus.CANCELLED,
                error="Task was cancelled before execution.",
                started_at=None,
                finished_at=now,
            )
        )
        return True

    async def wait_until_idle(self) -> None:
        """Wait until queued and active tasks have finished."""

        while True:
            await asyncio.sleep(0)

            if self._queue.is_empty and self.active_task_count == 0:
                return

    async def stop(self, cancel_running: bool = False) -> None:
        """Stop workers gracefully or cancel active executions."""

        if not isinstance(cancel_running, bool):
            raise TypeError("cancel_running must be a boolean.")

        async with self._state_lock:
            if not self._started:
                return

            self._stopping = True

            if cancel_running:
                for execution in tuple(self._running_tasks.values()):
                    execution.cancel()

                await asyncio.gather(
                    *tuple(self._running_tasks.values()),
                    return_exceptions=True,
                )
            else:
                await self.wait_until_idle()

            workers = tuple(self._workers)

            for worker in workers:
                worker.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

            self._workers.clear()
            self._running_tasks.clear()
            self._started = False

    async def _worker_loop(self) -> None:
        """Continuously retrieve and execute queued tasks."""

        while True:
            if self._stopping and self._queue.is_empty:
                return

            try:
                task = await self._queue.get()
            except BackgroundTaskNotFoundError:
                await asyncio.sleep(0.01)
                continue

            execution = asyncio.create_task(
                self._execute_task(task),
                name=f"seven-background-task-{task.task_id}",
            )
            self._running_tasks[task.task_id] = execution

            try:
                await execution
            except asyncio.CancelledError:
                current_worker = asyncio.current_task()

                if current_worker is not None and current_worker.cancelling():
                    if not execution.done():
                        execution.cancel()

                    await asyncio.gather(
                        execution,
                        return_exceptions=True,
                    )
                    raise

                await self._notify_cancelled(task)
            finally:
                self._running_tasks.pop(task.task_id, None)

    async def _execute_task(self, task: BackgroundTask) -> None:
        """Execute a task and deliver its result."""

        result = await self._executor.execute(task)
        await self._notify_result(result)

    async def _notify_cancelled(self, task: BackgroundTask) -> None:
        """Deliver a cancellation result."""

        now = datetime.now(timezone.utc)

        await self._notify_result(
            BackgroundTaskResult(
                task_id=task.task_id,
                status=BackgroundTaskStatus.CANCELLED,
                error="Task execution was cancelled.",
                started_at=now,
                finished_at=now,
            )
        )

    async def _notify_result(self, result: BackgroundTaskResult) -> None:
        """Deliver a task result to the configured callback."""

        if self._result_handler is None:
            return

        callback_result = self._result_handler(result)

        if inspect.isawaitable(callback_result):
            await callback_result
