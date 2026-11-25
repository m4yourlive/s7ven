
import asyncio

import pytest

from seven.core.integrations.background.exceptions import (
    BackgroundTaskError,
)
from seven.core.integrations.background.executor import (
    BackgroundTaskEngine,
)
from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskStatus,
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


def create_pool(
    worker_count: int = 2,
    result_handler=None,
) -> BackgroundWorkerPool:
    return BackgroundWorkerPool(
        task_queue=BackgroundTaskQueue(),
        executor=BackgroundTaskEngine(
            lambda action, parameters: {"executed": action}
        ),
        worker_count=worker_count,
        result_handler=result_handler,
    )


def test_worker_pool_rejects_invalid_queue() -> None:
    with pytest.raises(TypeError):
        BackgroundWorkerPool(
            None,  # type: ignore[arg-type]
            BackgroundTaskEngine(lambda action, parameters: None),
        )


def test_worker_pool_rejects_invalid_executor() -> None:
    with pytest.raises(TypeError):
        BackgroundWorkerPool(
            BackgroundTaskQueue(),
            None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("worker_count", [0, -1])
def test_worker_pool_rejects_non_positive_worker_count(
    worker_count: int,
) -> None:
    with pytest.raises(ValueError):
        create_pool(worker_count=worker_count)


def test_worker_pool_rejects_boolean_worker_count() -> None:
    with pytest.raises(TypeError):
        create_pool(worker_count=True)  # type: ignore[arg-type]


def test_worker_pool_starts_workers() -> None:
    async def scenario() -> None:
        pool = create_pool(worker_count=3)

        await pool.start()

        assert pool.is_running
        assert pool.active_worker_count == 3

        await pool.stop()

        assert not pool.is_running
        assert pool.active_worker_count == 0

    asyncio.run(scenario())


def test_worker_pool_start_is_idempotent() -> None:
    async def scenario() -> None:
        pool = create_pool()

        await pool.start()
        first_workers = tuple(pool._workers)

        await pool.start()

        assert tuple(pool._workers) == first_workers

        await pool.stop()

    asyncio.run(scenario())


def test_worker_pool_executes_submitted_task() -> None:
    async def scenario() -> None:
        completed = asyncio.Event()
        results = []

        def handle_result(result) -> None:
            results.append(result)
            completed.set()

        pool = create_pool(result_handler=handle_result)

        await pool.start()
        task = create_task()

        await pool.submit(task)
        await asyncio.wait_for(completed.wait(), timeout=2)

        assert len(results) == 1
        assert results[0].task_id == task.task_id
        assert results[0].status is BackgroundTaskStatus.COMPLETED

        await pool.stop()

    asyncio.run(scenario())


def test_worker_pool_runs_tasks_concurrently() -> None:
    async def scenario() -> None:
        active = 0
        maximum_active = 0
        completed = asyncio.Event()
        results = []

        async def handler(action, parameters):
            nonlocal active, maximum_active

            active += 1
            maximum_active = max(maximum_active, active)

            await asyncio.sleep(0.05)

            active -= 1
            return action

        def handle_result(result) -> None:
            results.append(result)

            if len(results) == 4:
                completed.set()

        pool = BackgroundWorkerPool(
            task_queue=BackgroundTaskQueue(),
            executor=BackgroundTaskEngine(handler),
            worker_count=2,
            result_handler=handle_result,
        )

        await pool.start()

        for index in range(4):
            await pool.submit(create_task(f"Task {index}"))

        await asyncio.wait_for(completed.wait(), timeout=3)

        assert maximum_active == 2
        assert len(results) == 4

        await pool.stop()

    asyncio.run(scenario())


def test_worker_pool_stop_is_idempotent() -> None:
    async def scenario() -> None:
        pool = create_pool()

        await pool.start()
        await pool.stop()
        await pool.stop()

        assert not pool.is_running

    asyncio.run(scenario())


def test_worker_pool_rejects_submission_while_stopping() -> None:
    async def scenario() -> None:
        pool = create_pool()
        pool._stopping = True

        with pytest.raises(BackgroundTaskError):
            await pool.submit(create_task())

    asyncio.run(scenario())


def test_worker_pool_wait_until_idle() -> None:
    async def scenario() -> None:
        completed = asyncio.Event()

        def handle_result(result) -> None:
            completed.set()

        pool = create_pool(result_handler=handle_result)

        await pool.start()
        await pool.submit(create_task())

        await pool.wait_until_idle()

        assert completed.is_set()
        await pool.stop()

    asyncio.run(scenario())
