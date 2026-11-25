
import asyncio
from datetime import timezone
from uuid import uuid4

import pytest

from seven.core.integrations.background.executor import (
    BackgroundTaskEngine,
)
from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskResult,
    BackgroundTaskStatus,
)


def create_task() -> BackgroundTask:
    return BackgroundTask(
        name="Test Task",
        action="test.action",
        parameters={"value": 42},
    )


def test_engine_rejects_non_callable_handler() -> None:
    with pytest.raises(TypeError):
        BackgroundTaskEngine(None)  # type: ignore[arg-type]


def test_engine_rejects_invalid_task() -> None:
    engine = BackgroundTaskEngine(lambda action, parameters: None)

    with pytest.raises(TypeError):
        asyncio.run(engine.execute(None))  # type: ignore[arg-type]


def test_engine_executes_synchronous_handler() -> None:
    task = create_task()
    calls = []

    def handler(action: str, parameters: dict) -> dict:
        calls.append((action, parameters))
        return {"success": True}

    engine = BackgroundTaskEngine(handler)
    result = asyncio.run(engine.execute(task))

    assert isinstance(result, BackgroundTaskResult)
    assert result.task_id == task.task_id
    assert result.status is BackgroundTaskStatus.COMPLETED
    assert result.output == {"success": True}
    assert result.error is None
    assert len(calls) == 1
    assert calls[0][0] == task.action
    assert calls[0][1] == task.parameters
    assert calls[0][1] is not task.parameters


def test_engine_executes_asynchronous_handler() -> None:
    task = create_task()

    async def handler(action: str, parameters: dict) -> str:
        await asyncio.sleep(0)
        return f"Executed {action}: {parameters['value']}"

    engine = BackgroundTaskEngine(handler)
    result = asyncio.run(engine.execute(task))

    assert result.status is BackgroundTaskStatus.COMPLETED
    assert result.output == "Executed test.action: 42"


def test_engine_returns_failed_result_when_handler_raises() -> None:
    task = create_task()

    def handler(action: str, parameters: dict) -> None:
        raise RuntimeError("Execution failed.")

    engine = BackgroundTaskEngine(handler)
    result = asyncio.run(engine.execute(task))

    assert result.status is BackgroundTaskStatus.FAILED
    assert result.error == "Execution failed."
    assert result.output is None
    assert result.task_id == task.task_id


def test_engine_records_execution_timestamps() -> None:
    engine = BackgroundTaskEngine(lambda action, parameters: "done")

    result = asyncio.run(engine.execute(create_task()))

    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.started_at.tzinfo is not None
    assert result.finished_at.tzinfo is not None
    assert result.started_at.tzinfo == timezone.utc
    assert result.finished_at >= result.started_at


def test_engine_propagates_cancellation() -> None:
    task = create_task()

    async def handler(action: str, parameters: dict) -> None:
        raise asyncio.CancelledError

    engine = BackgroundTaskEngine(handler)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.execute(task))
    