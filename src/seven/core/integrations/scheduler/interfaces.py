
"""Contracts for scheduled task automation."""

from typing import Protocol, runtime_checkable
from uuid import UUID

from seven.core.integrations.scheduler.models import (
    ScheduledExecution,
    ScheduledTask,
)


@runtime_checkable
class Scheduler(Protocol):
    """Define the scheduler service contract."""

    async def schedule(self, task: ScheduledTask) -> UUID:
        """Register a scheduled task."""
        ...

    async def cancel(self, schedule_id: UUID) -> bool:
        """Cancel a scheduled task."""
        ...

    async def get(self, schedule_id: UUID) -> ScheduledTask | None:
        """Retrieve a scheduled task."""
        ...


@runtime_checkable
class ScheduledTaskHandler(Protocol):
    """Execute actions requested by scheduled tasks."""

    async def execute(
        self,
        action: str,
        parameters: dict,
    ) -> object:
        """Execute a scheduled action."""
        ...


@runtime_checkable
class ScheduledTaskStore(Protocol):
    """Persist scheduled tasks and execution history."""

    async def save_task(self, task: ScheduledTask) -> None:
        """Persist a scheduled task."""
        ...

    async def get_task(self, schedule_id: UUID) -> ScheduledTask | None:
        """Retrieve a scheduled task."""
        ...

    async def list_tasks(self) -> tuple[ScheduledTask, ...]:
        """Return all stored scheduled tasks."""
        ...

    async def delete_task(self, schedule_id: UUID) -> None:
        """Delete a stored scheduled task."""
        ...

    async def save_execution(
        self,
        execution: ScheduledExecution,
    ) -> None:
        """Persist an execution record."""
        ...

    async def get_execution_history(
        self,
        schedule_id: UUID,
    ) -> tuple[ScheduledExecution, ...]:
        """Retrieve execution history for a scheduled task."""
        ...
