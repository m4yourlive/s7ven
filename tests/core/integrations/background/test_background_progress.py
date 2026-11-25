
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from seven.core.integrations.background.models import (
    BackgroundTaskStatus,
)
from seven.core.integrations.background.progress import (
    BackgroundTaskProgressTracker,
    TaskProgress,
)


@pytest.fixture
def tracker() -> BackgroundTaskProgressTracker:
    return BackgroundTaskProgressTracker()


@pytest.fixture
def task_id():
    return uuid4()


def test_task_progress_validates_fields(task_id) -> None:
    progress = TaskProgress(
        task_id=task_id,
        status=BackgroundTaskStatus.RUNNING,
        percentage=25,
        message="Processing",
        updated_at=datetime.now(timezone.utc),
    )

    assert progress.task_id == task_id
    assert progress.percentage == 25


def test_task_progress_rejects_invalid_percentage(task_id) -> None:
    with pytest.raises(ValueError):
        TaskProgress(
            task_id=task_id,
            status=BackgroundTaskStatus.RUNNING,
            percentage=101,
        )


def test_task_progress_rejects_boolean_percentage(task_id) -> None:
    with pytest.raises(TypeError):
        TaskProgress(
            task_id=task_id,
            status=BackgroundTaskStatus.RUNNING,
            percentage=True,
        )


def test_task_progress_requires_timezone_aware_datetime(task_id) -> None:
    with pytest.raises(ValueError):
        TaskProgress(
            task_id=task_id,
            status=BackgroundTaskStatus.RUNNING,
            updated_at=datetime.now(),
        )


@pytest.mark.asyncio
async def test_register_creates_pending_progress(tracker, task_id) -> None:
    progress = await tracker.register(task_id)

    assert progress.task_id == task_id
    assert progress.status is BackgroundTaskStatus.PENDING
    assert progress.percentage == 0
    assert progress.updated_at is not None


@pytest.mark.asyncio
async def test_register_rejects_duplicate_task(tracker, task_id) -> None:
    await tracker.register(task_id)

    with pytest.raises(ValueError):
        await tracker.register(task_id)


@pytest.mark.asyncio
async def test_update_progress_changes_percentage(tracker, task_id) -> None:
    await tracker.register(task_id)
    progress = await tracker.update_progress(
        task_id,
        45,
        "Downloading",
    )

    assert progress.percentage == 45
    assert progress.message == "Downloading"


@pytest.mark.asyncio
async def test_update_progress_rejects_invalid_percentage(
    tracker,
    task_id,
) -> None:
    await tracker.register(task_id)

    with pytest.raises(ValueError):
        await tracker.update_progress(task_id, 120)


@pytest.mark.asyncio
async def test_update_progress_rejects_boolean_percentage(
    tracker,
    task_id,
) -> None:
    await tracker.register(task_id)

    with pytest.raises(TypeError):
        await tracker.update_progress(task_id, True)


@pytest.mark.asyncio
async def test_update_status_changes_task_status(
    tracker,
    task_id,
) -> None:
    await tracker.register(task_id)

    progress = await tracker.update_status(
        task_id,
        BackgroundTaskStatus.RUNNING,
        "Executing",
    )

    assert progress.status is BackgroundTaskStatus.RUNNING
    assert progress.message == "Executing"


@pytest.mark.asyncio
async def test_completed_status_sets_percentage_to_100(
    tracker,
    task_id,
) -> None:
    await tracker.register(task_id)
    await tracker.update_progress(task_id, 60)

    progress = await tracker.update_status(
        task_id,
        BackgroundTaskStatus.COMPLETED,
    )

    assert progress.status is BackgroundTaskStatus.COMPLETED
    assert progress.percentage == 100


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        BackgroundTaskStatus.COMPLETED,
        BackgroundTaskStatus.FAILED,
        BackgroundTaskStatus.CANCELLED,
    ],
)
async def test_finished_task_rejects_progress_updates(
    tracker,
    task_id,
    status,
) -> None:
    await tracker.register(task_id)
    await tracker.update_status(task_id, status)

    with pytest.raises(ValueError):
        await tracker.update_progress(task_id, 50)


@pytest.mark.asyncio
async def test_finished_task_rejects_status_updates(
    tracker,
    task_id,
) -> None:
    await tracker.register(task_id)
    await tracker.update_status(
        task_id,
        BackgroundTaskStatus.COMPLETED,
    )

    with pytest.raises(ValueError):
        await tracker.update_status(
            task_id,
            BackgroundTaskStatus.RUNNING,
        )


@pytest.mark.asyncio
async def test_get_returns_registered_progress(
    tracker,
    task_id,
) -> None:
    await tracker.register(task_id)

    progress = await tracker.get(task_id)

    assert progress is not None
    assert progress.task_id == task_id


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown_task(tracker) -> None:
    assert await tracker.get(uuid4()) is None


@pytest.mark.asyncio
async def test_list_all_returns_progress_snapshot(tracker) -> None:
    first_id = uuid4()
    second_id = uuid4()

    await tracker.register(first_id)
    await tracker.register(second_id)

    records = await tracker.list_all()

    assert len(records) == 2
    assert {record.task_id for record in records} == {
        first_id,
        second_id,
    }


@pytest.mark.asyncio
async def test_remove_deletes_progress(tracker, task_id) -> None:
    await tracker.register(task_id)

    assert await tracker.remove(task_id) is True
    assert await tracker.remove(task_id) is False
    assert await tracker.get(task_id) is None


@pytest.mark.asyncio
async def test_clear_removes_all_progress(tracker) -> None:
    first_id = uuid4()
    second_id = uuid4()

    await tracker.register(first_id)
    await tracker.register(second_id)

    removed = await tracker.clear()

    assert {record.task_id for record in removed} == {
        first_id,
        second_id,
    }
    assert await tracker.list_all() == ()


@pytest.mark.asyncio
async def test_tracker_rejects_invalid_task_id(tracker) -> None:
    with pytest.raises(TypeError):
        await tracker.get("invalid")  # type: ignore[arg-type]
