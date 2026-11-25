
from datetime import datetime, timedelta, timezone

import pytest

from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
)
from seven.core.integrations.scheduler.timing import ScheduleTiming


@pytest.fixture
def once_task():
    return ScheduledTask(
        name="Backup",
        action="filesystem.backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, 10, tzinfo=timezone.utc),
    )


@pytest.fixture
def interval_task():
    return ScheduledTask(
        name="Health check",
        action="system.health_check",
        schedule_type=ScheduleType.INTERVAL,
        run_at=datetime(2026, 12, 1, 10, tzinfo=timezone.utc),
        interval_seconds=60,
    )


def test_one_time_schedule_returns_initial_run(once_task):
    after = datetime(2026, 12, 1, 9, tzinfo=timezone.utc)

    assert ScheduleTiming.get_next_run(once_task, after) == once_task.run_at


def test_one_time_schedule_returns_none_after_execution(once_task):
    after = datetime(2026, 12, 1, 11, tzinfo=timezone.utc)

    assert ScheduleTiming.get_next_run(once_task, after) is None


def test_one_time_schedule_returns_none_when_already_run(once_task):
    after = datetime(2026, 12, 1, 9, tzinfo=timezone.utc)

    assert ScheduleTiming.get_next_run(
        once_task,
        after,
        run_count=1,
    ) is None


def test_interval_schedule_returns_initial_run(interval_task):
    after = datetime(2026, 12, 1, 9, tzinfo=timezone.utc)

    assert ScheduleTiming.get_next_run(interval_task, after) == (
        interval_task.run_at
    )


def test_interval_schedule_calculates_next_run(interval_task):
    after = interval_task.run_at + timedelta(seconds=30)

    assert ScheduleTiming.get_next_run(interval_task, after) == (
        interval_task.run_at + timedelta(seconds=60)
    )


def test_interval_schedule_skips_missed_intervals(interval_task):
    after = interval_task.run_at + timedelta(seconds=150)

    assert ScheduleTiming.get_next_run(interval_task, after) == (
        interval_task.run_at + timedelta(seconds=180)
    )


def test_interval_schedule_respects_max_runs(interval_task):
    object.__setattr__(interval_task, "max_runs", 3)

    after = interval_task.run_at + timedelta(seconds=60)

    assert ScheduleTiming.get_next_run(
        interval_task,
        after,
        run_count=3,
    ) is None


def test_timing_rejects_naive_datetime(once_task):
    with pytest.raises(ValueError, match="timezone-aware"):
        ScheduleTiming.get_next_run(
            once_task,
            datetime(2026, 12, 1),
        )


def test_timing_rejects_negative_run_count(once_task):
    after = datetime(2026, 12, 1, 9, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="negative"):
        ScheduleTiming.get_next_run(
            once_task,
            after,
            run_count=-1,
        )


def test_timing_rejects_invalid_run_count(once_task):
    after = datetime(2026, 12, 1, 9, tzinfo=timezone.utc)

    with pytest.raises(TypeError, match="integer"):
        ScheduleTiming.get_next_run(
            once_task,
            after,
            run_count=1.5,
        )


def test_timing_rejects_invalid_task():
    with pytest.raises(TypeError, match="ScheduledTask"):
        ScheduleTiming.get_next_run(
            object(),
            datetime(2026, 12, 1, tzinfo=timezone.utc),
        )
