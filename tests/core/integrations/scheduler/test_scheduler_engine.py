
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from seven.core.integrations.scheduler.engine import ScheduledTaskEngine
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
    ScheduledTaskStatus,
)


@pytest.fixture
def task():
    return ScheduledTask(
        name="Daily backup",
        action="filesystem.backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def validator():
    class Validator:
        def validate(self, task):
            if not isinstance(task, ScheduledTask):
                raise ValueError("Invalid scheduled task.")

    return Validator()


@pytest.mark.asyncio
async def test_engine_executes_scheduled_task(task, validator):
    handler = AsyncMock()
    handler.execute.return_value = "Backup completed"

    engine = ScheduledTaskEngine(handler, validator)

    result = await engine.execute(task)

    assert result.status is ScheduledTaskStatus.COMPLETED
    assert result.output == "Backup completed"
    assert result.error is None
    handler.execute.assert_awaited_once_with(
        task.action,
        task.parameters,
    )


@pytest.mark.asyncio
async def test_engine_records_execution_timestamps(task, validator):
    handler = AsyncMock()
    engine = ScheduledTaskEngine(handler, validator)

    result = await engine.execute(task)

    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.started_at.tzinfo is not None
    assert result.finished_at >= result.started_at


@pytest.mark.asyncio
async def test_engine_uses_custom_scheduled_time(task, validator):
    handler = AsyncMock()
    engine = ScheduledTaskEngine(handler, validator)

    scheduled_time = datetime(2026, 12, 2, tzinfo=timezone.utc)
    result = await engine.execute(task, scheduled_for=scheduled_time)

    assert result.scheduled_for == scheduled_time


@pytest.mark.asyncio
async def test_engine_records_handler_failure(task, validator):
    handler = AsyncMock()
    handler.execute.side_effect = RuntimeError("Backup failed")

    engine = ScheduledTaskEngine(handler, validator)

    result = await engine.execute(task)

    assert result.status is ScheduledTaskStatus.FAILED
    assert result.error == "Backup failed"
    assert result.finished_at is not None


@pytest.mark.asyncio
async def test_engine_persists_execution(task, validator):
    handler = AsyncMock()
    store = AsyncMock()

    engine = ScheduledTaskEngine(handler, validator, store)

    result = await engine.execute(task)

    store.save_execution.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_engine_does_not_execute_disabled_task(task, validator):
    object.__setattr__(task, "enabled", False)

    handler = AsyncMock()
    engine = ScheduledTaskEngine(handler, validator)

    with pytest.raises(ValueError, match="Disabled"):
        await engine.execute(task)

    handler.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_rejects_naive_scheduled_time(task, validator):
    handler = AsyncMock()
    engine = ScheduledTaskEngine(handler, validator)

    with pytest.raises(ValueError, match="timezone-aware"):
        await engine.execute(
            task,
            scheduled_for=datetime(2026, 12, 2),
        )

    handler.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_engine_propagates_task_cancellation(task, validator):
    handler = AsyncMock()
    handler.execute.side_effect = __import__("asyncio").CancelledError

    store = AsyncMock()
    engine = ScheduledTaskEngine(handler, validator, store)

    with pytest.raises(__import__("asyncio").CancelledError):
        await engine.execute(task)

    store.save_execution.assert_awaited_once()

    saved = store.save_execution.await_args.args[0]
    assert saved.status is ScheduledTaskStatus.CANCELLED
    assert saved.error == "Scheduled task execution was cancelled."


@pytest.mark.asyncio
async def test_engine_rejects_missing_dependencies():
    with pytest.raises(ValueError):
        ScheduledTaskEngine(None, object())

    with pytest.raises(ValueError):
        ScheduledTaskEngine(object(), None)
