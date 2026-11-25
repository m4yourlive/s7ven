
"""Tests for recurring task management."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from seven.core.integrations.scheduler.exceptions import (
    ScheduledTaskAlreadyExistsError,
    ScheduledTaskNotFoundError,
)
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
)
from seven.core.integrations.scheduler.recurrence import (
    RecurringTaskManager,
    RecurringTaskProgress,
)


@pytest.fixture
def manager() -> RecurringTaskManager:
    return RecurringTaskManager()


@pytest.fixture
def recurring_task() -> ScheduledTask:
    return ScheduledTask(
        name="Recurring Task",
        action="test.action",
        schedule_type=ScheduleType.INTERVAL,
        run_at=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        interval_seconds=60,
        max_runs=3,
    )


def test_register_recurring_task(manager, recurring_task):
    progress = manager.register(recurring_task)

    assert progress.schedule_id == recurring_task.schedule_id
    assert progress.run_count == 0
    assert progress.cancelled is False


def test_register_rejects_non_recurring_task(manager):
    task = ScheduledTask(
        name="One Time Task",
        action="test.action",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="Only interval tasks"):
        manager.register(task)


def test_register_rejects_disabled_task(manager, recurring_task):
    task = ScheduledTask(
        name=recurring_task.name,
        action=recurring_task.action,
        schedule_type=ScheduleType.INTERVAL,
        run_at=recurring_task.run_at,
        interval_seconds=recurring_task.interval_seconds,
        enabled=False,
    )

    with pytest.raises(ValueError, match="Disabled tasks"):
        manager.register(task)


def test_register_rejects_duplicate_task(manager, recurring_task):
    manager.register(recurring_task)

    with pytest.raises(ScheduledTaskAlreadyExistsError):
        manager.register(recurring_task)


def test_get_task_returns_registered_task(manager, recurring_task):
    manager.register(recurring_task)

    assert manager.get_task(recurring_task.schedule_id) is recurring_task


def test_get_progress_returns_initial_progress(manager, recurring_task):
    manager.register(recurring_task)

    progress = manager.get_progress(recurring_task.schedule_id)

    assert progress.run_count == 0
    assert progress.cancelled is False


def test_get_next_run_returns_initial_run(manager, recurring_task):
    manager.register(recurring_task)

    next_run = manager.get_next_run(
        recurring_task.schedule_id,
        datetime(2026, 11, 25, 11, 59, tzinfo=UTC),
    )

    assert next_run == recurring_task.run_at


def test_record_execution_increments_run_count(manager, recurring_task):
    manager.register(recurring_task)

    progress = manager.record_execution(recurring_task.schedule_id)

    assert progress.run_count == 1
    assert progress.cancelled is False


def test_record_execution_tracks_multiple_runs(manager, recurring_task):
    manager.register(recurring_task)

    manager.record_execution(recurring_task.schedule_id)
    progress = manager.record_execution(recurring_task.schedule_id)

    assert progress.run_count == 2


def test_record_execution_rejects_max_runs(manager, recurring_task):
    manager.register(recurring_task)

    for _ in range(3):
        manager.record_execution(recurring_task.schedule_id)

    with pytest.raises(ValueError, match="maximum run count"):
        manager.record_execution(recurring_task.schedule_id)


def test_get_next_run_returns_none_after_max_runs(
    manager,
    recurring_task,
):
    manager.register(recurring_task)

    for _ in range(3):
        manager.record_execution(recurring_task.schedule_id)

    next_run = manager.get_next_run(
        recurring_task.schedule_id,
        datetime(2026, 11, 25, 12, 10, tzinfo=UTC),
    )

    assert next_run is None


def test_get_next_run_uses_execution_count(manager, recurring_task):
    manager.register(recurring_task)
    manager.record_execution(recurring_task.schedule_id)

    next_run = manager.get_next_run(
        recurring_task.schedule_id,
        recurring_task.run_at,
    )

    assert next_run == recurring_task.run_at + timedelta(seconds=60)


def test_cancel_recurring_task(manager, recurring_task):
    manager.register(recurring_task)

    assert manager.cancel(recurring_task.schedule_id) is True
    assert manager.get_progress(recurring_task.schedule_id).cancelled is True


def test_cancel_returns_false_when_already_cancelled(
    manager,
    recurring_task,
):
    manager.register(recurring_task)

    manager.cancel(recurring_task.schedule_id)

    assert manager.cancel(recurring_task.schedule_id) is False


def test_cancelled_task_has_no_next_run(manager, recurring_task):
    manager.register(recurring_task)
    manager.cancel(recurring_task.schedule_id)

    next_run = manager.get_next_run(
        recurring_task.schedule_id,
        datetime(2026, 11, 25, 11, 59, tzinfo=UTC),
    )

    assert next_run is None


def test_cancelled_task_rejects_execution_record(
    manager,
    recurring_task,
):
    manager.register(recurring_task)
    manager.cancel(recurring_task.schedule_id)

    with pytest.raises(ValueError, match="Cancelled tasks"):
        manager.record_execution(recurring_task.schedule_id)


def test_unregister_removes_task(manager, recurring_task):
    manager.register(recurring_task)

    assert manager.unregister(recurring_task.schedule_id) is True

    with pytest.raises(ScheduledTaskNotFoundError):
        manager.get_task(recurring_task.schedule_id)


def test_unknown_task_raises_not_found(manager):
    with pytest.raises(ScheduledTaskNotFoundError):
        manager.get_progress(uuid4())


def test_invalid_schedule_id_is_rejected(manager):
    with pytest.raises(TypeError, match="schedule_id"):
        manager.get_progress("invalid")


def test_progress_rejects_negative_run_count():
    with pytest.raises(ValueError, match="cannot be negative"):
        RecurringTaskProgress(
            schedule_id=uuid4(),
            run_count=-1,
        )
