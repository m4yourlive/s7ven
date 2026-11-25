
"""Tests for scheduler lifecycle management."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from seven.core.integrations.scheduler.exceptions import (
    ScheduledTaskAlreadyExistsError,
    ScheduledTaskNotFoundError,
)
from seven.core.integrations.scheduler.manager import SchedulerManager
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledExecution,
    ScheduledTask,
    ScheduledTaskStatus,
)


@pytest.fixture
def task():
    return ScheduledTask(
        name="Backup Task",
        action="backup.create",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def store():
    instance = Mock()
    instance.list_tasks = AsyncMock(return_value=())
    instance.get_task = AsyncMock(return_value=None)
    instance.save_task = AsyncMock()
    instance.delete_task = AsyncMock(return_value=True)
    return instance


@pytest.fixture
def engine():
    instance = Mock()
    instance.execute = AsyncMock(
        return_value=ScheduledExecution(
            schedule_id=uuid4(),
            scheduled_for=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
            status=ScheduledTaskStatus.COMPLETED,
        )
    )
    return instance


@pytest.fixture
def manager(engine, store):
    return SchedulerManager(engine=engine, store=store)


@pytest.mark.asyncio
async def test_scheduler_starts(manager):
    await manager.start()

    assert manager.is_running is True


@pytest.mark.asyncio
async def test_scheduler_start_is_idempotent(manager, store):
    await manager.start()
    await manager.start()

    assert store.list_tasks.await_count == 1


@pytest.mark.asyncio
async def test_scheduler_restores_enabled_tasks(
    manager,
    store,
    task,
):
    store.list_tasks.return_value = (
        task,
        ScheduledTask(
            name="Disabled Task",
            action="backup.create",
            schedule_type=ScheduleType.ONCE,
            run_at=task.run_at,
            enabled=False,
        ),
    )

    await manager.start()

    assert task.schedule_id in manager._tasks
    assert len(manager._tasks) == 1


@pytest.mark.asyncio
async def test_scheduler_shutdown(manager):
    await manager.start()
    await manager.shutdown()

    assert manager.is_running is False


@pytest.mark.asyncio
async def test_scheduler_shutdown_is_idempotent(manager):
    await manager.start()
    await manager.shutdown()
    await manager.shutdown()

    assert manager.is_running is False


@pytest.mark.asyncio
async def test_register_persists_task(manager, store, task):
    await manager.start()

    result = await manager.register(task)

    assert result == task.schedule_id
    store.save_task.assert_awaited_once_with(task)
    assert task.schedule_id in manager._tasks


@pytest.mark.asyncio
async def test_register_while_stopped_persists_task(
    manager,
    store,
    task,
):
    await manager.register(task)

    store.save_task.assert_awaited_once_with(task)
    assert task.schedule_id not in manager._tasks


@pytest.mark.asyncio
async def test_register_rejects_duplicate_task(
    manager,
    store,
    task,
):
    await manager.start()
    await manager.register(task)

    with pytest.raises(ScheduledTaskAlreadyExistsError):
        await manager.register(task)


@pytest.mark.asyncio
async def test_register_rejects_existing_persisted_task(
    manager,
    store,
    task,
):
    store.get_task.return_value = task

    with pytest.raises(ScheduledTaskAlreadyExistsError):
        await manager.register(task)


@pytest.mark.asyncio
async def test_execute_requires_running_scheduler(
    manager,
    task,
):
    with pytest.raises(RuntimeError, match="not running"):
        await manager.execute(task.schedule_id)


@pytest.mark.asyncio
async def test_execute_requires_registered_task(
    manager,
):
    await manager.start()

    with pytest.raises(ScheduledTaskNotFoundError):
        await manager.execute(uuid4())


@pytest.mark.asyncio
async def test_execute_invokes_engine(
    manager,
    engine,
    task,
):
    await manager.start()
    await manager.register(task)

    result = await manager.execute(
        task.schedule_id,
        scheduled_for=task.run_at,
    )

    engine.execute.assert_awaited_once_with(
        task,
        scheduled_for=task.run_at,
    )
    assert result.status is ScheduledTaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_unregister_removes_task(
    manager,
    store,
    task,
):
    await manager.start()
    await manager.register(task)

    assert await manager.unregister(task.schedule_id) is True
    assert task.schedule_id not in manager._tasks
    store.delete_task.assert_awaited_once_with(task.schedule_id)


@pytest.mark.asyncio
async def test_unregister_missing_task_raises(
    manager,
    store,
):
    await manager.start()

    with pytest.raises(ScheduledTaskNotFoundError):
        await manager.unregister(uuid4())


@pytest.mark.asyncio
async def test_get_task_returns_persisted_task(
    manager,
    store,
    task,
):
    store.get_task.return_value = task

    assert await manager.get_task(task.schedule_id) is task


@pytest.mark.asyncio
async def test_list_tasks_returns_persisted_tasks(
    manager,
    store,
    task,
):
    store.list_tasks.return_value = (task,)

    assert await manager.list_tasks() == (task,)


@pytest.mark.asyncio
async def test_invalid_schedule_id_is_rejected(manager):
    with pytest.raises(TypeError, match="schedule_id"):
        await manager.get_task("invalid")


@pytest.mark.asyncio
async def test_shutdown_cancels_active_execution(
    manager,
    engine,
    store,
    task,
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_execute(*args, **kwargs):
        started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    engine.execute.side_effect = slow_execute

    await manager.start()
    await manager.register(task)

    execution = asyncio.create_task(
        manager.execute(task.schedule_id)
    )

    await started.wait()
    await manager.shutdown()

    assert cancelled.is_set()
    assert manager.is_running is False

    with pytest.raises(asyncio.CancelledError):
        await execution
