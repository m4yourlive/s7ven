
"""Background task execution engine."""

import inspect
from datetime import datetime, timezone
from typing import Any, Callable

from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskResult,
    BackgroundTaskStatus,
)


class BackgroundTaskEngine:
    """Execute background tasks and capture their results."""

    def __init__(
        self,
        action_handler: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        if not callable(action_handler):
            raise TypeError("action_handler must be callable.")

        self._action_handler = action_handler

    async def execute(
        self,
        task: BackgroundTask,
    ) -> BackgroundTaskResult:
        """Execute a task and return its result."""

        if not isinstance(task, BackgroundTask):
            raise TypeError("task must be a BackgroundTask.")

        started_at = datetime.now(timezone.utc)

        try:
            output = self._action_handler(
                task.action,
                task.parameters.copy(),
            )

            if inspect.isawaitable(output):
                output = await output

            finished_at = datetime.now(timezone.utc)

            return BackgroundTaskResult(
                task_id=task.task_id,
                status=BackgroundTaskStatus.COMPLETED,
                output=output,
                started_at=started_at,
                finished_at=finished_at,
            )

        except Exception as exc:
            finished_at = datetime.now(timezone.utc)

            return BackgroundTaskResult(
                task_id=task.task_id,
                status=BackgroundTaskStatus.FAILED,
                error=str(exc),
                started_at=started_at,
                finished_at=finished_at,
            )
