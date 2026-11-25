
"""Background task contracts."""

from typing import Protocol, runtime_checkable
from uuid import UUID

from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskResult,
)


@runtime_checkable
class BackgroundTaskExecutor(Protocol):
    """Execute a background task."""

    async def execute(
        self,
        task: BackgroundTask,
    ) -> BackgroundTaskResult:
        """Execute the supplied task."""


@runtime_checkable
class BackgroundTaskStore(Protocol):
    """Persist background tasks and results."""

    async def save_task(self, task: BackgroundTask) -> None:
        """Persist a background task."""

    async def get_task(
        self,
        task_id: UUID,
    ) -> BackgroundTask | None:
        """Retrieve a background task."""

    async def list_tasks(self) -> tuple[BackgroundTask, ...]:
        """List persisted background tasks."""

    async def save_result(
        self,
        result: BackgroundTaskResult,
    ) -> None:
        """Persist a task result."""

    async def get_result(
        self,
        task_id: UUID,
    ) -> BackgroundTaskResult | None:
        """Retrieve a task result."""
