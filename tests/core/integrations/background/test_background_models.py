
"""Tests for background task domain models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from seven.core.integrations.background.models import (
    BackgroundTask,
    BackgroundTaskResult,
    BackgroundTaskStatus,
)


def test_background_task_creation():
    task = BackgroundTask(
        name="Backup",
        action="backup.create",
    )

    assert task.name == "Backup"
    assert task.action == "backup.create"
    assert task.priority == 0
    assert task.parameters == {}


def test_background_task_normalizes_strings():
    task = BackgroundTask(
        name="  Backup  ",
        action="  backup.create  ",
    )

    assert task.name == "Backup"
    assert task.action == "backup.create"


def test_background_task_generates_unique_ids():
    first = BackgroundTask(name="First", action="test")
    second = BackgroundTask(name="Second", action="test")

    assert first.task_id != second.task_id


def test_background_task_preserves_parameters():
    parameters = {"path": "backup"}

    task = BackgroundTask(
        name="Backup",
        action="backup.create",
        parameters=parameters,
    )

    parameters["path"] = "modified"

    assert task.parameters == {"path": "backup"}


def test_background_task_rejects_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        BackgroundTask(name=" ", action="test")


def test_background_task_rejects_invalid_name_type():
    with pytest.raises(TypeError, match="name must be a string"):
        BackgroundTask(name=123, action="test")


def test_background_task_rejects_empty_action():
    with pytest.raises(ValueError, match="action cannot be empty"):
        BackgroundTask(name="Test", action=" ")


def test_background_task_rejects_invalid_action_type():
    with pytest.raises(TypeError, match="action must be a string"):
        BackgroundTask(name="Test", action=123)


def test_background_task_rejects_invalid_parameters():
    with pytest.raises(TypeError, match="parameters must be a dictionary"):
        BackgroundTask(
            name="Test",
            action="test",
            parameters=[],
        )


def test_background_task_rejects_invalid_id():
    with pytest.raises(TypeError, match="Task ID"):
        BackgroundTask(
            name="Test",
            action="test",
            task_id="invalid",
        )


def test_background_task_rejects_boolean_priority():
    with pytest.raises(TypeError, match="priority must be an integer"):
        BackgroundTask(
            name="Test",
            action="test",
            priority=True,
        )


def test_background_task_rejects_naive_creation_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        BackgroundTask(
            name="Test",
            action="test",
            created_at=datetime(2026, 1, 1),
        )


def test_background_task_result_creation():
    task_id = uuid4()

    result = BackgroundTaskResult(
        task_id=task_id,
        status=BackgroundTaskStatus.COMPLETED,
        output={"success": True},
    )

    assert result.task_id == task_id
    assert result.status is BackgroundTaskStatus.COMPLETED


def test_background_task_result_rejects_invalid_id():
    with pytest.raises(TypeError, match="Task ID"):
        BackgroundTaskResult(
            task_id="invalid",
            status=BackgroundTaskStatus.COMPLETED,
        )


def test_background_task_result_rejects_invalid_status():
    with pytest.raises(TypeError, match="Task status"):
        BackgroundTaskResult(
            task_id=uuid4(),
            status="completed",
        )


def test_background_task_result_rejects_invalid_error():
    with pytest.raises(ValueError, match="error cannot be empty"):
        BackgroundTaskResult(
            task_id=uuid4(),
            status=BackgroundTaskStatus.FAILED,
            error=" ",
        )


def test_background_task_result_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        BackgroundTaskResult(
            task_id=uuid4(),
            status=BackgroundTaskStatus.RUNNING,
            started_at=datetime(2026, 1, 1),
        )


def test_background_task_result_accepts_timestamps():
    now = datetime.now(UTC)

    result = BackgroundTaskResult(
        task_id=uuid4(),
        status=BackgroundTaskStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )

    assert result.started_at == now
    assert result.finished_at == now
