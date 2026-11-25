
"""Scheduler lifecycle and execution management."""

import asyncio
import inspect
from datetime import datetime, timezone
from uuid import UUID

from seven.core.integrations.scheduler.engine import ScheduledTaskEngine
from seven.core.integrations.scheduler.exceptions import (
    ScheduledTaskAlreadyExistsError,
    ScheduledTaskNotFoundError,
)
from seven.core.integrations.scheduler.interfaces import (
    ScheduledTaskStore,
)
from seven.core.integrations.scheduler.models import (
    ScheduledTask,
    ScheduledTaskStatus,
)


class SchedulerManager:
    """Coordinate scheduled task lifecycle and execution."""

    def __init__(
        self,
        engine: ScheduledTaskEngine,
        store: ScheduledTaskStore,
    ) -> None:
        if engine is None:
            raise TypeError("engine is required.")

        if store is None:
            raise TypeError("store is required.")

        self._engine = engine
        self._store = store
        self._running = False
        self._tasks: dict[UUID, ScheduledTask] = {}
        self._active_executions: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler is active."""

        return self._running

    async def start(self) -> None:
        """Restore enabled tasks and activate the scheduler."""

        async with self._lock:
            if self._running:
                return

            tasks = await self._store.list_tasks()

            self._tasks = {
                task.schedule_id: task
                for task in tasks
                if task.enabled
            }

            self._running = True

    async def shutdown(self, cancel_running: bool = True) -> None:
        """Stop the scheduler and optionally cancel active executions."""

        if not isinstance(cancel_running, bool):
            raise TypeError("cancel_running must be a boolean.")

        async with self._lock:
            if not self._running:
                return

            self._running = False
            active = tuple(self._active_executions)

            if cancel_running:
                for execution in active:
                    execution.cancel()

        if active:
            await asyncio.gather(*active, return_exceptions=True)

        async with self._lock:
            self._active_executions.clear()

    async def register(self, task: ScheduledTask) -> UUID:
        """Persist and register an enabled scheduled task."""

        if not isinstance(task, ScheduledTask):
            raise TypeError("task must be a ScheduledTask.")

        async with self._lock:
            if task.schedule_id in self._tasks:
                raise ScheduledTaskAlreadyExistsError(
                    f"Task '{task.schedule_id}' is already registered."
                )

            existing = await self._store.get_task(task.schedule_id)

            if existing is not None:
                raise ScheduledTaskAlreadyExistsError(
                    f"Task '{task.schedule_id}' already exists."
                )

            await self._store.save_task(task)

            if self._running and task.enabled:
                self._tasks[task.schedule_id] = task

        return task.schedule_id

    async def unregister(self, schedule_id: UUID) -> bool:
        """Remove a task from the scheduler and persistent storage."""

        self._validate_id(schedule_id)

        async with self._lock:
            if schedule_id not in self._tasks:
                existing = await self._store.get_task(schedule_id)

                if existing is None:
                    raise ScheduledTaskNotFoundError(
                        f"Task '{schedule_id}' was not found."
                    )

            self._tasks.pop(schedule_id, None)
            return await self._store.delete_task(schedule_id)

    async def get_task(self, schedule_id: UUID) -> ScheduledTask | None:
        """Retrieve a scheduled task."""

        self._validate_id(schedule_id)

        async with self._lock:
            return await self._store.get_task(schedule_id)

    async def list_tasks(self) -> tuple[ScheduledTask, ...]:
        """List all persisted scheduled tasks."""

        return await self._store.list_tasks()

    async def execute(
        self,
        schedule_id: UUID,
        scheduled_for: datetime | None = None,
    ):
        """Execute a registered scheduled task."""

        self._validate_id(schedule_id)

        async with self._lock:
            if not self._running:
                raise RuntimeError("Scheduler is not running.")

            task = self._tasks.get(schedule_id)

            if task is None:
                raise ScheduledTaskNotFoundError(
                    f"Task '{schedule_id}' is not registered."
                )

            if not task.enabled:
                raise ValueError("Scheduled task is disabled.")

            execution_task = asyncio.create_task(
                self._engine.execute(
                    task,
                    scheduled_for=scheduled_for,
                )
            )

            self._active_executions.add(execution_task)

        try:
            return await execution_task
        finally:
            async with self._lock:
                self._active_executions.discard(execution_task)

    @staticmethod
    def _validate_id(schedule_id: UUID) -> None:
        if not isinstance(schedule_id, UUID):
            raise TypeError("schedule_id must be a UUID.")
