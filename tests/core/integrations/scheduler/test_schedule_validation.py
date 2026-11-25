
from datetime import datetime, timezone

import pytest

from seven.core.integrations.scheduler.exceptions import (
    ScheduleValidationError,
)
from seven.core.integrations.scheduler.models import (
    ScheduleType,
    ScheduledTask,
)
from seven.core.integrations.scheduler.validation import (
    DefaultScheduleValidator,
)


@pytest.fixture
def validator():
    return DefaultScheduleValidator()


@pytest.fixture
def one_time_task():
    return ScheduledTask(
        name="Backup",
        action="filesystem.backup",
        schedule_type=ScheduleType.ONCE,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )


def test_validator_accepts_one_time_task(validator, one_time_task):
    validator.validate(one_time_task)


def test_validator_accepts_interval_task(validator):
    task = ScheduledTask(
        name="Health check",
        action="system.health_check",
        schedule_type=ScheduleType.INTERVAL,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        interval_seconds=60,
    )

    validator.validate(task)


def test_validator_rejects_invalid_task_type(validator):
    with pytest.raises(ScheduleValidationError):
        validator.validate(object())


def test_validator_rejects_empty_task_name(validator, one_time_task):
    object.__setattr__(one_time_task, "name", "")

    with pytest.raises(ScheduleValidationError, match="name"):
        validator.validate(one_time_task)


def test_validator_rejects_empty_action(validator, one_time_task):
    object.__setattr__(one_time_task, "action", "")

    with pytest.raises(ScheduleValidationError, match="action"):
        validator.validate(one_time_task)


def test_validator_rejects_invalid_schedule_type(
    validator,
    one_time_task,
):
    object.__setattr__(one_time_task, "schedule_type", "invalid")

    with pytest.raises(ScheduleValidationError, match="schedule type"):
        validator.validate(one_time_task)


def test_validator_rejects_naive_datetime(validator, one_time_task):
    object.__setattr__(
        one_time_task,
        "run_at",
        datetime(2026, 12, 1),
    )

    with pytest.raises(ScheduleValidationError, match="timezone-aware"):
        validator.validate(one_time_task)


def test_validator_rejects_invalid_parameters(validator, one_time_task):
    object.__setattr__(one_time_task, "parameters", [])

    with pytest.raises(ScheduleValidationError, match="dictionary"):
        validator.validate(one_time_task)


def test_validator_rejects_interval_without_duration(
    validator,
    one_time_task,
):
    task = ScheduledTask.__new__(ScheduledTask)

    object.__setattr__(task, "name", "Backup")
    object.__setattr__(task, "action", "backup")
    object.__setattr__(task, "schedule_type", ScheduleType.INTERVAL)
    object.__setattr__(
        task,
        "run_at",
        datetime(2026, 12, 1, tzinfo=timezone.utc),
    )
    object.__setattr__(task, "parameters", {})
    object.__setattr__(task, "interval_seconds", None)
    object.__setattr__(task, "max_runs", None)
    object.__setattr__(task, "schedule_id", one_time_task.schedule_id)
    object.__setattr__(task, "enabled", True)

    with pytest.raises(ScheduleValidationError, match="require an interval"):
        validator.validate(task)


def test_validator_rejects_non_positive_interval(validator):
    task = ScheduledTask(
        name="Health check",
        action="health.check",
        schedule_type=ScheduleType.INTERVAL,
        run_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        interval_seconds=1,
    )
    object.__setattr__(task, "interval_seconds", 0)

    with pytest.raises(ScheduleValidationError, match="greater than zero"):
        validator.validate(task)


def test_validator_rejects_non_positive_max_runs(
    validator,
    one_time_task,
):
    object.__setattr__(one_time_task, "max_runs", 0)

    with pytest.raises(ScheduleValidationError, match="Maximum runs"):
        validator.validate(one_time_task)
