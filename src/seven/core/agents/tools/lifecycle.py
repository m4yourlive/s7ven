"""Execution lifecycle tracking and metrics for agent tools."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from seven.core.agents.tools.context import ToolExecutionContext
from seven.core.agents.tools.models import ToolCall


class ToolExecutionStatus(StrEnum):
    """Represent the lifecycle status of a tool execution."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionRecord:
    """Store the lifecycle information of one tool execution."""

    execution_id: UUID
    call_id: UUID
    tool_name: str
    status: ToolExecutionStatus
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionMetrics:
    """Provide an immutable snapshot of tool execution metrics."""

    total_executions: int = 0
    succeeded: int = 0
    failed: int = 0
    timed_out: int = 0
    cancelled: int = 0
    active: int = 0
    total_duration_seconds: float = 0.0
    average_duration_seconds: float = 0.0


class ToolExecutionTracker:
    """Track tool execution records and aggregate metrics."""

    def __init__(self) -> None:
        self._records: dict[UUID, ToolExecutionRecord] = {}

    @property
    def records(self) -> tuple[ToolExecutionRecord, ...]:
        """Return all execution records."""
        return tuple(self._records.values())

    @property
    def metrics(self) -> ToolExecutionMetrics:
        """Calculate an immutable metrics snapshot."""
        records = tuple(self._records.values())

        succeeded = sum(
            record.status == ToolExecutionStatus.SUCCEEDED
            for record in records
        )
        failed = sum(
            record.status == ToolExecutionStatus.FAILED
            for record in records
        )
        timed_out = sum(
            record.status == ToolExecutionStatus.TIMED_OUT
            for record in records
        )
        cancelled = sum(
            record.status == ToolExecutionStatus.CANCELLED
            for record in records
        )
        active = sum(
            record.status == ToolExecutionStatus.RUNNING
            for record in records
        )

        durations = [
            record.duration_seconds
            for record in records
            if record.duration_seconds is not None
        ]

        total_duration = sum(durations)

        return ToolExecutionMetrics(
            total_executions=len(records),
            succeeded=succeeded,
            failed=failed,
            timed_out=timed_out,
            cancelled=cancelled,
            active=active,
            total_duration_seconds=total_duration,
            average_duration_seconds=(
                total_duration / len(durations)
                if durations
                else 0.0
            ),
        )

    def get_record(
        self,
        execution_id: UUID,
    ) -> ToolExecutionRecord | None:
        """Retrieve an execution record by its ID."""
        if not isinstance(execution_id, UUID):
            raise TypeError("execution_id must be a UUID.")

        return self._records.get(execution_id)

    def start(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionRecord:
        """Create a record when a tool execution starts."""
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall instance.")

        if not isinstance(context, ToolExecutionContext):
            raise TypeError(
                "context must be a ToolExecutionContext instance."
            )

        if context.execution_id in self._records:
            raise ValueError("Execution ID is already being tracked.")

        record = ToolExecutionRecord(
            execution_id=context.execution_id,
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

        self._records[record.execution_id] = record

        return record

    def finish(
        self,
        execution_id: UUID,
        status: ToolExecutionStatus,
        error: str | None = None,
    ) -> ToolExecutionRecord:
        """Complete an active execution record."""
        if not isinstance(execution_id, UUID):
            raise TypeError("execution_id must be a UUID.")

        if not isinstance(status, ToolExecutionStatus):
            raise TypeError("status must be a ToolExecutionStatus.")

        if status == ToolExecutionStatus.RUNNING:
            raise ValueError("An execution cannot finish with RUNNING status.")

        if error is not None and not isinstance(error, str):
            raise TypeError("error must be a string or None.")

        record = self._records.get(execution_id)

        if record is None:
            raise KeyError(f"Execution '{execution_id}' was not found.")

        if record.status != ToolExecutionStatus.RUNNING:
            raise ValueError("Execution has already finished.")

        if status == ToolExecutionStatus.SUCCEEDED and error:
            raise ValueError("A successful execution cannot contain an error.")

        if status != ToolExecutionStatus.SUCCEEDED and not error:
            raise ValueError("A failed execution must contain an error.")

        completed_at = datetime.now(UTC)
        duration = max(
            0.0,
            (completed_at - record.started_at).total_seconds(),
        )

        updated_record = ToolExecutionRecord(
            execution_id=record.execution_id,
            call_id=record.call_id,
            tool_name=record.tool_name,
            status=status,
            started_at=record.started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            error=error,
        )

        self._records[execution_id] = updated_record

        return updated_record

    def clear(self) -> None:
        """Remove all execution records and reset metrics."""
        self._records.clear()