
import asyncio

import pytest

from seven.core.integrations.background import (
    BackgroundTask,
    BackgroundTaskCancellationManager,
    BackgroundTaskEngine,
    BackgroundTaskProgressTracker,
    BackgroundTaskQueue,
    BackgroundTaskStatus,
    BackgroundWorkerPool,
)


def create_task(name: str = "Integration Task") -> BackgroundTask:
    return BackgroundTask(
        name=name,
        action="integration.execute",
        parameters={"value": 42},
    )


@pytest.mark.asyncio
async def test_complete_background_task_lifecycle() -> None:
    queue = BackgroundTaskQueue()
    tracker = BackgroundTaskProgressTracker()
    results = []

    async def handler(action, parameters):
        await asyncio.sleep(0.01)
        return {"action": action, "value": parameters["value"]}

    async def result_handler(result):
        results.append(result)

        progress = await tracker.get(result.task_id)

        if progress is not None:
            await tracker.update_status(
                result.task_id,
                result.status,
                result.error,
            )

    engine = BackgroundTaskEngine(handler)
    workers = BackgroundWorkerPool(
        task_queue=queue,
        executor=engine,
        worker_count=2,
        result_handler=result_handler,
    )

    task = create_task()

    await tracker.register(task.task_id)
    await workers.start()
    await workers.submit(task)
    await workers.wait_until_idle()

    progress = await tracker.get(task.task_id)

    assert len(results) == 1
    assert results[0].status is BackgroundTaskStatus.COMPLETED
    assert results[0].output == {
        "action": "integration.execute",
        "value": 42,
    }
    assert progress is not None
    assert progress.status is BackgroundTaskStatus.COMPLETED

    await workers.stop()


@pytest.mark.asyncio
async def test_failed_task_updates_progress() -> None:
    tracker = BackgroundTaskProgressTracker()
    results = []

    def handler(action, parameters):
        raise RuntimeError("Integration failure.")

    async def result_handler(result):
        results.append(result)
        await tracker.update_status(
            result.task_id,
            result.status,
            result.error,
        )

    workers = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(handler),
        result_handler=result_handler,
    )

    task = create_task()

    await tracker.register(task.task_id)
    await workers.start()
    await workers.submit(task)
    await workers.wait_until_idle()

    progress = await tracker.get(task.task_id)

    assert len(results) == 1
    assert results[0].status is BackgroundTaskStatus.FAILED
    assert progress is not None
    assert progress.status is BackgroundTaskStatus.FAILED
    assert progress.message == "Integration failure."

    await workers.stop()


@pytest.mark.asyncio
async def test_queued_cancellation_updates_progress() -> None:
    queue = BackgroundTaskQueue()
    tracker = BackgroundTaskProgressTracker()

    workers = BackgroundWorkerPool(
        queue,
        BackgroundTaskEngine(lambda action, parameters: None),
    )
    cancellation = BackgroundTaskCancellationManager(
        workers,
        tracker,
    )

    task = create_task()

    await tracker.register(task.task_id)
    await workers.submit(task)

    assert await cancellation.cancel(task.task_id) is True

    progress = await tracker.get(task.task_id)

    assert progress is not None
    assert progress.status is BackgroundTaskStatus.CANCELLED
    assert await queue.contains(task.task_id) is False


@pytest.mark.asyncio
async def test_running_cancellation_updates_progress() -> None:
    started = asyncio.Event()
    tracker = BackgroundTaskProgressTracker()
    results = []

    async def handler(action, parameters):
        started.set()
        await asyncio.Event().wait()

    async def result_handler(result):
        results.append(result)

    workers = BackgroundWorkerPool(
        BackgroundTaskQueue(),
        BackgroundTaskEngine(handler),
        result_handler=result_handler,
    )
    cancellation = BackgroundTaskCancellationManager(
        workers,
        tracker,
    )

    task = create_task()

    await tracker.register(task.task_id)
    await tracker.update_status(
        task.task_id,
        BackgroundTaskStatus.RUNNING,
    )

    await workers.start()
    await workers.submit(task)
    await asyncio.wait_for(started.wait(), timeout=2)

    assert await cancellation.cancel(task.task_id) is True

    progress = await tracker.get(task.task_id)

    assert progress is not None
    assert progress.status is BackgroundTaskStatus.CANCELLED
    assert len(results) == 1
    assert results[0].status is BackgroundTaskStatus.CANCELLED

    await workers.stop()


def test_public_api_exports() -> None:
    from seven.core.integrations.background import (
        BackgroundTaskCancellationManager,
        BackgroundTaskEngine,
        BackgroundTaskProgressTracker,
        BackgroundTaskQueue,
        BackgroundWorkerPool,
    )

    assert BackgroundTaskCancellationManager is not None
    assert BackgroundTaskEngine is not None
    assert BackgroundTaskProgressTracker is not None
    assert BackgroundTaskQueue is not None
    assert BackgroundWorkerPool is not None
