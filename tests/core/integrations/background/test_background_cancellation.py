
import asyncio

import pytest

from seven.core.integrations.background.cancellation import (
    BackgroundTaskCancellationManager,
)
from seven.core.integrations.background.executor import (
    BackgroundTaskEngine,
)
from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskStatus,
)
from seven.core.integrations.background.progress import (
    BackgroundTaskProgressTracker,
)
from seven.core.integrations.background.queue import (
    BackgroundTaskQueue,
)
from seven.core.integrations.background.worker import (
    BackgroundWorkerPool,
)


def create_task(name: str = "Test Task") -> BackgroundTask:
    return BackgroundTask(
        name=name,
        action="test.action",
    )


def test_cancellation_manager_rejects_invalid_worker_pool() -> None:
    with pytest.raises(TypeError):
        BackgroundTaskCancellationManager(None)  # type: ignore[arg-type]


def test_cancellation_manager_rejects_invalid_tracker() -> None:
    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(lambda action, parameters: None),
    )

    with pytest.raises(TypeError):
        BackgroundTaskCancellationManager(
            pool,
            progress_tracker=None,  # type: ignore[arg-type]
        ) if False else BackgroundTaskCancellationManager(
            pool,
            progress_tracker="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_cancel_queued_task() -> None:
    queue = BackgroundTaskQueue()
    pool = BackgroundWorkerPool(
        queue,
        BackgroundTaskEngine(lambda action, parameters: None),
    )
    tracker = BackgroundTaskProgressTracker()
    manager = BackgroundTaskCancellationManager(pool, tracker)
    task = create_task()

    await tracker.register(task.task_id)
    await pool.submit(task)

    assert await manager.cancel(task.task_id) is True
    assert await queue.contains(task.task_id) is False

    progress = await tracker.get(task.task_id)
    assert progress is not None
    assert progress.status is BackgroundTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_unknown_task_returns_false() -> None:
    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(lambda action, parameters: None),
    )
    manager = BackgroundTaskCancellationManager(pool)

    assert await manager.cancel(create_task().task_id) is False


@pytest.mark.asyncio
async def test_cancel_running_task() -> None:
    started = asyncio.Event()
    results = []

    async def handler(action, parameters):
        started.set()
        await asyncio.Event().wait()

    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(handler),
        worker_count=1,
        result_handler=results.append,
    )
    manager = BackgroundTaskCancellationManager(pool)
    task = create_task()

    await pool.start()
    await pool.submit(task)
    await asyncio.wait_for(started.wait(), timeout=2)

    assert await manager.cancel(task.task_id) is True

    assert len(results) == 1
    assert results[0].status is BackgroundTaskStatus.CANCELLED

    await pool.stop()


@pytest.mark.asyncio
async def test_worker_continues_after_task_cancellation() -> None:
    started = asyncio.Event()
    results = []

    async def handler(action, parameters):
        if parameters.get("block"):
            started.set()
            await asyncio.Event().wait()
        return "completed"

    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(handler),
        worker_count=1,
        result_handler=results.append,
    )

    first = BackgroundTask(
        name="Blocking Task",
        action="test.action",
        parameters={"block": True},
    )
    second = create_task("Second Task")

    await pool.start()
    await pool.submit(first)
    await asyncio.wait_for(started.wait(), timeout=2)

    assert await pool.cancel_task(first.task_id) is True

    await pool.submit(second)

    for _ in range(100):
        if any(result.task_id == second.task_id for result in results):
            break
        await asyncio.sleep(0.01)

    assert any(
        result.task_id == first.task_id
        and result.status is BackgroundTaskStatus.CANCELLED
        for result in results
    )
    assert any(
        result.task_id == second.task_id
        and result.status is BackgroundTaskStatus.COMPLETED
        for result in results
    )

    await pool.stop()


@pytest.mark.asyncio
async def test_graceful_shutdown_waits_for_running_tasks() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    results = []

    async def handler(action, parameters):
        started.set()
        await release.wait()
        return "completed"

    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(handler),
        result_handler=results.append,
    )

    await pool.start()
    await pool.submit(create_task())
    await asyncio.wait_for(started.wait(), timeout=2)

    shutdown = asyncio.create_task(pool.stop())
    await asyncio.sleep(0.02)

    assert not shutdown.done()

    release.set()
    await asyncio.wait_for(shutdown, timeout=2)

    assert len(results) == 1
    assert results[0].status is BackgroundTaskStatus.COMPLETED
    assert not pool.is_running


@pytest.mark.asyncio
async def test_immediate_shutdown_cancels_running_tasks() -> None:
    started = asyncio.Event()
    results = []

    async def handler(action, parameters):
        started.set()
        await asyncio.Event().wait()

    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(handler),
        result_handler=results.append,
    )

    await pool.start()
    await pool.submit(create_task())
    await asyncio.wait_for(started.wait(), timeout=2)

    await asyncio.wait_for(
        pool.stop(cancel_running=True),
        timeout=2,
    )

    assert len(results) == 1
    assert results[0].status is BackgroundTaskStatus.CANCELLED
    assert not pool.is_running


@pytest.mark.asyncio
async def test_shutdown_manager_delegates_to_worker_pool() -> None:
    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(lambda action, parameters: None),
    )
    manager = BackgroundTaskCancellationManager(pool)

    await pool.start()
    await manager.shutdown()

    assert not pool.is_running


@pytest.mark.asyncio
async def test_cancel_rejects_invalid_task_id() -> None:
    pool = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(lambda action, parameters: None),
    )
    manager = BackgroundTaskCancellationManager(pool)

    with pytest.raises(TypeError):
        await manager.cancel("invalid")  # type: ignore[arg-type]
