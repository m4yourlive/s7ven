"""Tests for tool execution lifecycle tracking."""

from uuid import uuid4

import pytest

from seven.core.agents.tools import (
    ToolCall,
    ToolExecutionContext,
    ToolExecutionStatus,
    ToolExecutionTracker,
)


def create_execution():
    call = ToolCall(tool_name="test_tool")
    context = ToolExecutionContext()

    return call, context


def test_tracker_starts_empty():
    tracker = ToolExecutionTracker()

    assert tracker.records == ()
    assert tracker.metrics.total_executions == 0
    assert tracker.metrics.active == 0


def test_start_creates_running_record():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    record = tracker.start(call, context)

    assert record.status == ToolExecutionStatus.RUNNING
    assert record.execution_id == context.execution_id
    assert record.call_id == call.call_id
    assert record.tool_name == call.tool_name
    assert record.completed_at is None
    assert record.duration_seconds is None


def test_start_rejects_duplicate_execution_id():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    with pytest.raises(ValueError):
        tracker.start(call, context)


def test_start_rejects_invalid_call():
    tracker = ToolExecutionTracker()

    with pytest.raises(TypeError):
        tracker.start(object(), ToolExecutionContext())


def test_start_rejects_invalid_context():
    tracker = ToolExecutionTracker()

    with pytest.raises(TypeError):
        tracker.start(ToolCall(tool_name="test_tool"), object())


def test_finish_marks_execution_as_succeeded():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    record = tracker.finish(
        context.execution_id,
        ToolExecutionStatus.SUCCEEDED,
    )

    assert record.status == ToolExecutionStatus.SUCCEEDED
    assert record.completed_at is not None
    assert record.duration_seconds is not None
    assert record.error is None


def test_finish_marks_execution_as_failed():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    record = tracker.finish(
        context.execution_id,
        ToolExecutionStatus.FAILED,
        error="Execution failed.",
    )

    assert record.status == ToolExecutionStatus.FAILED
    assert record.error == "Execution failed."


def test_finish_rejects_running_status():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    with pytest.raises(ValueError):
        tracker.finish(
            context.execution_id,
            ToolExecutionStatus.RUNNING,
        )


def test_finish_rejects_missing_error_for_failure():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    with pytest.raises(ValueError):
        tracker.finish(
            context.execution_id,
            ToolExecutionStatus.FAILED,
        )


def test_finish_rejects_error_for_success():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    with pytest.raises(ValueError):
        tracker.finish(
            context.execution_id,
            ToolExecutionStatus.SUCCEEDED,
            error="Unexpected error.",
        )


def test_finish_rejects_unknown_execution():
    tracker = ToolExecutionTracker()

    with pytest.raises(KeyError):
        tracker.finish(
            uuid4(),
            ToolExecutionStatus.SUCCEEDED,
        )


def test_finish_rejects_already_finished_execution():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)
    tracker.finish(
        context.execution_id,
        ToolExecutionStatus.SUCCEEDED,
    )

    with pytest.raises(ValueError):
        tracker.finish(
            context.execution_id,
            ToolExecutionStatus.FAILED,
            error="Late failure.",
        )


def test_get_record_returns_existing_record():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    assert tracker.get_record(context.execution_id) is not None


def test_get_record_returns_none_for_unknown_id():
    tracker = ToolExecutionTracker()

    assert tracker.get_record(uuid4()) is None


def test_get_record_rejects_invalid_id():
    tracker = ToolExecutionTracker()

    with pytest.raises(TypeError):
        tracker.get_record("invalid")


def test_metrics_track_active_executions():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)

    assert tracker.metrics.total_executions == 1
    assert tracker.metrics.active == 1
    assert tracker.metrics.succeeded == 0


def test_metrics_track_successful_executions():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)
    tracker.finish(
        context.execution_id,
        ToolExecutionStatus.SUCCEEDED,
    )

    assert tracker.metrics.total_executions == 1
    assert tracker.metrics.succeeded == 1
    assert tracker.metrics.active == 0


def test_metrics_track_failed_executions():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)
    tracker.finish(
        context.execution_id,
        ToolExecutionStatus.FAILED,
        error="Failed.",
    )

    assert tracker.metrics.failed == 1


def test_metrics_track_timeout_executions():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)
    tracker.finish(
        context.execution_id,
        ToolExecutionStatus.TIMED_OUT,
        error="Timed out.",
    )

    assert tracker.metrics.timed_out == 1


def test_metrics_track_cancelled_executions():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)
    tracker.finish(
        context.execution_id,
        ToolExecutionStatus.CANCELLED,
        error="Cancelled.",
    )

    assert tracker.metrics.cancelled == 1


def test_metrics_calculate_average_duration():
    tracker = ToolExecutionTracker()

    for _ in range(2):
        call, context = create_execution()
        tracker.start(call, context)
        tracker.finish(
            context.execution_id,
            ToolExecutionStatus.SUCCEEDED,
        )

    metrics = tracker.metrics

    assert metrics.total_duration_seconds >= 0
    assert metrics.average_duration_seconds >= 0


def test_clear_removes_records_and_metrics():
    tracker = ToolExecutionTracker()
    call, context = create_execution()

    tracker.start(call, context)
    tracker.finish(
        context.execution_id,
        ToolExecutionStatus.SUCCEEDED,
    )

    tracker.clear()

    assert tracker.records == ()
    assert tracker.metrics.total_executions == 0


def test_metrics_snapshot_is_immutable():
    tracker = ToolExecutionTracker()

    with pytest.raises(AttributeError):
        tracker.metrics.succeeded = 10