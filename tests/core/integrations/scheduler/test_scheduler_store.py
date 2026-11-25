
"""Tests for SQLite scheduled task persistence."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledExecution,
    ScheduledTask,
    ScheduledTaskStatus,
)
from seven.core.integrations.scheduler.store import (
    SQLiteScheduledTaskStore,
)


@pytest.fixture
def store(tmp_path):
    """Create a synchronous store fixture."""

    instance = SQLiteScheduledTaskStore(
        tmp_path / "scheduler.db"
    )

    yield instance

    if instance._connection is not None:
        instance._connection.close()
        instance._connection = None


@pytest.fixture
def task():
    return ScheduledTask(
        name="Recurring Backup",
        action="backup.create",
        schedule_type=ScheduleType.INTERVAL,
        run_at=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        parameters={"directory": "backups"},
        interval_seconds=3600,
        max_runs=5,
    )


@pytest.fixture
def execution(task):
    return ScheduledExecution(
        schedule_id=task.schedule_id,
        scheduled_for=task.run_at,
        status=ScheduledTaskStatus.COMPLETED,
        started_at=task.run_at,
        finished_at=task.run_at,
        output={"result": "success"},
    )


@pytest.mark.asyncio
async def test_save_and_get_task(store, task):
    await store.initialize()
    await store.save_task(task)

    result = await store.get_task(task.schedule_id)

    assert result == task


@pytest.mark.asyncio
async def test_get_missing_task_returns_none(store):
    await store.initialize()

    assert await store.get_task(uuid4()) is None


@pytest.mark.asyncio
async def test_list_tasks(store, task):
    await store.initialize()
    await store.save_task(task)

    tasks = await store.list_tasks()

    assert len(tasks) == 1
    assert tasks[0] == task


@pytest.mark.asyncio
async def test_save_task_updates_existing_task(store, task):
    await store.initialize()
    await store.save_task(task)

    updated = ScheduledTask(
        name="Updated Backup",
        action=task.action,
        schedule_type=task.schedule_type,
        run_at=task.run_at,
        parameters={"directory": "new-backups"},
        interval_seconds=task.interval_seconds,
        max_runs=task.max_runs,
        schedule_id=task.schedule_id,
    )

    await store.save_task(updated)

    result = await store.get_task(task.schedule_id)

    assert result == updated
    assert len(await store.list_tasks()) == 1


@pytest.mark.asyncio
async def test_delete_task(store, task):
    await store.initialize()
    await store.save_task(task)

    assert await store.delete_task(task.schedule_id) is True
    assert await store.get_task(task.schedule_id) is None


@pytest.mark.asyncio
async def test_delete_missing_task_returns_false(store):
    await store.initialize()

    assert await store.delete_task(uuid4()) is False


@pytest.mark.asyncio
async def test_save_and_get_execution(store, task, execution):
    await store.initialize()
    await store.save_task(task)
    await store.save_execution(execution)

    history = await store.get_execution_history(task.schedule_id)

    assert len(history) == 1
    assert history[0] == execution


@pytest.mark.asyncio
async def test_execution_history_respects_limit(store, task):
    await store.initialize()
    await store.save_task(task)

    for _ in range(3):
        await store.save_execution(
            ScheduledExecution(
                schedule_id=task.schedule_id,
                scheduled_for=task.run_at,
                status=ScheduledTaskStatus.COMPLETED,
            )
        )

    history = await store.get_execution_history(
        task.schedule_id,
        limit=2,
    )

    assert len(history) == 2


@pytest.mark.asyncio
async def test_execution_history_is_empty_for_unknown_task(store):
    await store.initialize()

    assert await store.get_execution_history(uuid4()) == ()


@pytest.mark.asyncio
async def test_store_requires_initialization():
    store = SQLiteScheduledTaskStore()

    with pytest.raises(RuntimeError, match="initialized"):
        await store.list_tasks()


@pytest.mark.asyncio
async def test_save_task_rejects_invalid_object(store):
    await store.initialize()

    with pytest.raises(TypeError, match="ScheduledTask"):
        await store.save_task("invalid")


@pytest.mark.asyncio
async def test_save_execution_rejects_invalid_object(store):
    await store.initialize()

    with pytest.raises(TypeError, match="ScheduledExecution"):
        await store.save_execution("invalid")


@pytest.mark.asyncio
async def test_invalid_schedule_id_is_rejected(store):
    await store.initialize()

    with pytest.raises(TypeError, match="schedule_id"):
        await store.get_task("invalid")


@pytest.mark.asyncio
async def test_execution_history_rejects_invalid_limit(store):
    await store.initialize()

    with pytest.raises(ValueError, match="positive"):
        await store.get_execution_history(uuid4(), limit=0)


@pytest.mark.asyncio
async def test_close_allows_reinitialization(tmp_path):
    database_path = tmp_path / "scheduler.db"
    store = SQLiteScheduledTaskStore(database_path)

    await store.initialize()
    await store.close()
    await store.initialize()

    assert await store.list_tasks() == ()

    await store.close()


@pytest.mark.asyncio
async def test_persistence_survives_reopening(tmp_path, task):
    database_path = tmp_path / "scheduler.db"

    first_store = SQLiteScheduledTaskStore(database_path)
    await first_store.initialize()
    await first_store.save_task(task)
    await first_store.close()

    second_store = SQLiteScheduledTaskStore(database_path)
    await second_store.initialize()

    restored = await second_store.get_task(task.schedule_id)

    assert restored == task

    await second_store.close()
