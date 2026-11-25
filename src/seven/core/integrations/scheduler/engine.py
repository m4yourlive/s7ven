
"""Execution engine for scheduled tasks."""

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from seven.core.integrations.scheduler.exceptions import (
    ScheduleValidationError,
)
from seven.core.integrations.scheduler.models import (
    ScheduledExecution,
    ScheduledTask,
    ScheduledTaskStatus,
)


class ScheduledTaskEngine:
    """Execute scheduled tasks and persist their execution results."""

    def __init__(
        self,
        action_handler: Any,
        validator: Any,
        execution_store: Any | None = None,
    ) -> None:
        if action_handler is None:
            raise ValueError("Scheduled task handler is required.")

        if validator is None:
            raise ValueError("Schedule validator is required.")

        self._action_handler = action_handler
        self._validator = validator
        self._execution_store = execution_store

    async def execute(
        self,
        task: ScheduledTask,
        scheduled_for: datetime | None = None,
    ) -> ScheduledExecution:
        """Execute a scheduled task and return its execution result."""

        validation_result = self._validator.validate(task)

        if inspect.isawaitable(validation_result):
            await validation_result

        if not task.enabled:
            raise ScheduleValidationError(
                "Disabled scheduled tasks cannot be executed."
            )

        scheduled_time = scheduled_for or task.run_at

        if not isinstance(scheduled_time, datetime):
            raise TypeError("scheduled_for must be a datetime.")

        if (
            scheduled_time.tzinfo is None
            or scheduled_time.utcoffset() is None
        ):
            raise ValueError("scheduled_for must be timezone-aware.")

        execution_id = uuid4()
        started_at = datetime.now(timezone.utc)

        try:
            output = self._action_handler.execute(
                task.action,
                task.parameters,
            )

            if inspect.isawaitable(output):
                output = await output

            execution = ScheduledExecution(
                schedule_id=task.schedule_id,
                scheduled_for=scheduled_time,
                execution_id=execution_id,
                status=ScheduledTaskStatus.COMPLETED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                output=output,
            )

        except asyncio.CancelledError:
            execution = ScheduledExecution(
                schedule_id=task.schedule_id,
                scheduled_for=scheduled_time,
                execution_id=execution_id,
                status=ScheduledTaskStatus.CANCELLED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error="Scheduled task execution was cancelled.",
            )

            await self._save_execution(execution)
            raise

        except Exception as error:
            execution = ScheduledExecution(
                schedule_id=task.schedule_id,
                scheduled_for=scheduled_time,
                execution_id=execution_id,
                status=ScheduledTaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=str(error),
            )

        await self._save_execution(execution)
        return execution

    async def _save_execution(
        self,
        execution: ScheduledExecution,
    ) -> None:
        """Persist an execution record when a store is configured."""

        if self._execution_store is None:
            return

        result = self._execution_store.save_execution(execution)

        if inspect.isawaitable(result):
            await result
