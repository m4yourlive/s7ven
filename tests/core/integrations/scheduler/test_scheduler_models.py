
from datetime import datetime, timezone

import pytest

from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledExecution,
    ScheduledTask,
    ScheduledTaskStatus,
)


def test_creates_one_time_task():
    task = ScheduledTask(
        name="Backup",
        action="filesystem.backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )

    assert task.name == "Backup"
    assert task.schedule_type is ScheduleType.ONCE
    assert task.interval_seconds is None


def test_creates_interval_task():
    task = ScheduledTask(
        name="Health check",
        action="system.health_check",
        schedule_type=ScheduleType.INTERVAL,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        interval_seconds=60,
    )

    assert task.interval_seconds == 60


def test_normalizes_task_strings():
    task = ScheduledTask(
        name="  Backup  ",
        action="  filesystem.backup  ",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )

    assert task.name == "Backup"
    assert task.action == "filesystem.backup"


def test_copies_task_parameters():
    parameters = {"path": "C:/Backup"}

    task = ScheduledTask(
        name="Backup",
        action="filesystem.backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        parameters=parameters,
    )

    assert task.parameters == parameters
    assert task.parameters is not parameters


def test_rejects_empty_task_name():
    with pytest.raises(ValueError):
        ScheduledTask(
            name=" ",
            action="backup",
            schedule_type=ScheduleType.ONCE,
            run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )


def test_rejects_empty_task_action():
    with pytest.raises(ValueError):
        ScheduledTask(
            name="Backup",
            action=" ",
            schedule_type=ScheduleType.ONCE,
            run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )


def test_rejects_naive_datetime():
    with pytest.raises(ValueError):
        ScheduledTask(
            name="Backup",
            action="backup",
            schedule_type=ScheduleType.ONCE,
            run_at=datetime(2026, 12, 1),
        )


def test_rejects_interval_without_duration():
    with pytest.raises(ValueError):
        ScheduledTask(
            name="Backup",
            action="backup",
            schedule_type=ScheduleType.INTERVAL,
            run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )


def test_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        ScheduledTask(
            name="Backup",
            action="backup",
            schedule_type=ScheduleType.INTERVAL,
            run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
            interval_seconds=0,
        )


def test_rejects_multiple_runs_for_one_time_schedule():
    with pytest.raises(ValueError):
        ScheduledTask(
            name="Backup",
            action="backup",
            schedule_type=ScheduleType.ONCE,
            run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
            max_runs=2,
        )


def test_creates_scheduled_execution():
    task = ScheduledTask(
        name="Backup",
        action="backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )

    execution = ScheduledExecution(
        schedule_id=task.schedule_id,
        scheduled_for=task.run_at,
        status=ScheduledTaskStatus.COMPLETED,
    )

    assert execution.schedule_id == task.schedule_id
    assert execution.status is ScheduledTaskStatus.COMPLETED
