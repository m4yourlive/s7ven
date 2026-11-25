
"""Tests for background task queue."""

from uuid import uuid4

import pytest

from seven.core.integrations.background.exceptions import (
    BackgroundTaskAlreadyExistsError,
    BackgroundTaskNotFoundError,
    BackgroundTaskQueueFullError,
)
from seven.core.integrations.background.models import BackgroundTask
from seven.core.integrations.background.queue import BackgroundTaskQueue


@pytest.fixture
def queue():
    return BackgroundTaskQueue(max_size=3)


def create_task(
    name: str,
    priority: int = 0,
) -> BackgroundTask:
    return BackgroundTask(
        name=name,
        action="test.execute",
        priority=priority,
    )


@pytest.mark.asyncio
async def test_submit_task(queue):
    task = create_task("Task")

    result = await queue.submit(task)

    assert result == task.task_id
    assert queue.size == 1


@pytest.mark.asyncio
async def test_submit_rejects_invalid_task(queue):
    with pytest.raises(TypeError, match="BackgroundTask"):
        await queue.submit("invalid")


@pytest.mark.asyncio
async def test_submit_rejects_duplicate_task(queue):
    task = create_task("Task")

    await queue.submit(task)

    with pytest.raises(BackgroundTaskAlreadyExistsError):
        await queue.submit(task)


@pytest.mark.asyncio
async def test_submit_rejects_full_queue(queue):
    for index in range(3):
        await queue.submit(create_task(f"Task {index}"))

    with pytest.raises(BackgroundTaskQueueFullError):
        await queue.submit(create_task("Overflow"))


@pytest.mark.asyncio
async def test_get_returns_pending_task(queue):
    task = create_task("Task")
    await queue.submit(task)

    result = await queue.get()

    assert result == task
    assert queue.size == 0


@pytest.mark.asyncio
async def test_get_respects_priority(queue):
    low = create_task("Low", priority=1)
    high = create_task("High", priority=10)
    medium = create_task("Medium", priority=5)

    await queue.submit(low)
    await queue.submit(high)
    await queue.submit(medium)

    assert await queue.get() == high
    assert await queue.get() == medium
    assert await queue.get() == low


@pytest.mark.asyncio
async def test_get_preserves_fifo_for_equal_priority(queue):
    first = create_task("First", priority=5)
    second = create_task("Second", priority=5)

    await queue.submit(first)
    await queue.submit(second)

    assert await queue.get() == first
    assert await queue.get() == second


@pytest.mark.asyncio
async def test_get_rejects_empty_queue(queue):
    with pytest.raises(BackgroundTaskNotFoundError):
        await queue.get()


@pytest.mark.asyncio
async def test_remove_pending_task(queue):
    task = create_task("Task")
    await queue.submit(task)

    assert await queue.remove(task.task_id) is True
    assert queue.size == 0


@pytest.mark.asyncio
async def test_remove_missing_task_returns_false(queue):
    assert await queue.remove(uuid4()) is False


@pytest.mark.asyncio
async def test_removed_task_is_not_returned(queue):
    task = create_task("Task")

    await queue.submit(task)
    await queue.remove(task.task_id)

    with pytest.raises(BackgroundTaskNotFoundError):
        await queue.get()


@pytest.mark.asyncio
async def test_contains_pending_task(queue):
    task = create_task("Task")
    await queue.submit(task)

    assert await queue.contains(task.task_id) is True


@pytest.mark.asyncio
async def test_contains_missing_task(queue):
    assert await queue.contains(uuid4()) is False


@pytest.mark.asyncio
async def test_snapshot_returns_priority_order(queue):
    low = create_task("Low", priority=1)
    high = create_task("High", priority=10)
    medium = create_task("Medium", priority=5)

    await queue.submit(low)
    await queue.submit(high)
    await queue.submit(medium)

    snapshot = await queue.snapshot()

    assert snapshot == (high, medium, low)
    assert queue.size == 3


@pytest.mark.asyncio
async def test_clear_removes_all_tasks(queue):
    first = create_task("First")
    second = create_task("Second")

    await queue.submit(first)
    await queue.submit(second)

    removed = await queue.clear()

    assert {task.task_id for task in removed} == {
        first.task_id,
        second.task_id,
    }
    assert queue.is_empty
    assert queue.size == 0


def test_queue_rejects_invalid_max_size():
    with pytest.raises(TypeError, match="max_size"):
        BackgroundTaskQueue(max_size=True)


def test_queue_rejects_non_positive_max_size():
    with pytest.raises(ValueError, match="positive"):
        BackgroundTaskQueue(max_size=0)


@pytest.mark.asyncio
async def test_queue_rejects_invalid_task_id(queue):
    with pytest.raises(TypeError, match="task_id"):
        await queue.contains("invalid")


@pytest.mark.asyncio
async def test_queue_capacity_recovers_after_get(queue):
    for index in range(3):
        await queue.submit(create_task(f"Task {index}"))

    await queue.get()

    task = create_task("New Task")
    await queue.submit(task)

    assert queue.size == 3
    assert queue.is_full is True


@pytest.mark.asyncio
async def test_queue_capacity_recovers_after_remove(queue):
    tasks = [
        create_task(f"Task {index}")
        for index in range(3)
    ]

    for task in tasks:
        await queue.submit(task)

    await queue.remove(tasks[0].task_id)

    await queue.submit(create_task("Replacement"))

    assert queue.size == 3
