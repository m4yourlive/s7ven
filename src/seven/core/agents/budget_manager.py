
"""Track and enforce autonomous agent execution budgets."""

import math
import time
from uuid import UUID, uuid4

from seven.core.agents.budget import (
    ExecutionBudget,
    ExecutionBudgetExceededError,
    ExecutionBudgetSnapshot,
    ExecutionLease,
)


class ExecutionBudgetManager:
    """Track resource usage and enforce execution limits."""

    def __init__(
        self,
        budget: ExecutionBudget | None = None,
        task_id: UUID | None = None,
    ) -> None:
        if budget is not None and not isinstance(budget, ExecutionBudget):
            raise TypeError("budget must be an ExecutionBudget instance.")

        if task_id is not None and not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID or None.")

        self._budget = budget or ExecutionBudget()
        self._task_id = task_id
        self._started_at = time.monotonic()
        self._tool_calls_started = 0
        self._active_leases: dict[UUID, ExecutionLease] = {}
        self._total_tool_execution_seconds = 0.0

    @property
    def budget(self) -> ExecutionBudget:
        return self._budget

    @property
    def task_id(self) -> UUID | None:
        return self._task_id

    @property
    def active_tool_calls(self) -> int:
        return len(self._active_leases)

    @property
    def tool_calls_started(self) -> int:
        return self._tool_calls_started

    def check_can_start_tool_call(self) -> None:
        """Raise when another tool call would exceed a resource limit."""
        elapsed = self._elapsed_seconds()

        if elapsed >= self._budget.max_execution_seconds:
            raise ExecutionBudgetExceededError(
                "Maximum task execution time exceeded."
            )

        if self._tool_calls_started >= self._budget.max_tool_calls:
            raise ExecutionBudgetExceededError(
                "Maximum tool call count exceeded."
            )

        if (
            self.active_tool_calls
            >= self._budget.max_concurrent_tool_calls
        ):
            raise ExecutionBudgetExceededError(
                "Maximum concurrent tool calls exceeded."
            )

        if (
            self._total_tool_execution_seconds
            >= self._budget.max_total_tool_execution_seconds
        ):
            raise ExecutionBudgetExceededError(
                "Maximum total tool execution time exceeded."
            )

    def start_tool_call(self, tool_name: str) -> ExecutionLease:
        """Reserve resources and register a new tool execution."""
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        self.check_can_start_tool_call()

        lease = ExecutionLease(
            lease_id=uuid4(),
            tool_name=tool_name,
            started_at=time.monotonic(),
        )

        self._active_leases[lease.lease_id] = lease
        self._tool_calls_started += 1

        return lease

    def complete_tool_call(self, lease: ExecutionLease) -> float:
        """Record a completed tool execution and return its duration."""
        if not isinstance(lease, ExecutionLease):
            raise TypeError("lease must be an ExecutionLease instance.")

        active_lease = self._active_leases.get(lease.lease_id)

        if active_lease != lease:
            raise ValueError("Execution lease is not active.")

        duration = max(0.0, time.monotonic() - lease.started_at)

        del self._active_leases[lease.lease_id]
        self._total_tool_execution_seconds += duration

        if duration > self._budget.max_tool_execution_seconds:
            raise ExecutionBudgetExceededError(
                f"Tool execution time exceeded for '{lease.tool_name}'."
            )

        if (
            self._total_tool_execution_seconds
            > self._budget.max_total_tool_execution_seconds
        ):
            raise ExecutionBudgetExceededError(
                "Maximum total tool execution time exceeded."
            )

        if self._elapsed_seconds() > self._budget.max_execution_seconds:
            raise ExecutionBudgetExceededError(
                "Maximum task execution time exceeded."
            )

        return duration

    def get_tool_timeout(self) -> float:
        """Return the maximum safe timeout for the next tool execution."""
        self.check_can_start_tool_call()

        remaining_task_time = (
            self._budget.max_execution_seconds - self._elapsed_seconds()
        )
        remaining_total_tool_time = (
            self._budget.max_total_tool_execution_seconds
            - self._total_tool_execution_seconds
        )

        timeout = min(
            self._budget.max_tool_execution_seconds,
            remaining_task_time,
            remaining_total_tool_time,
        )

        if not math.isfinite(timeout) or timeout <= 0:
            raise ExecutionBudgetExceededError(
                "No execution time remains for another tool call."
            )

        return timeout

    def snapshot(self) -> ExecutionBudgetSnapshot:
        """Return an immutable snapshot of current resource usage."""
        elapsed = self._elapsed_seconds()

        remaining_calls = max(
            0,
            self._budget.max_tool_calls - self._tool_calls_started,
        )
        remaining_execution = max(
            0.0,
            self._budget.max_execution_seconds - elapsed,
        )
        remaining_tool_execution = max(
            0.0,
            self._budget.max_total_tool_execution_seconds
            - self._total_tool_execution_seconds,
        )

        exhausted = (
            remaining_calls == 0
            or remaining_execution <= 0
            or remaining_tool_execution <= 0
            or self.active_tool_calls
            >= self._budget.max_concurrent_tool_calls
        )

        return ExecutionBudgetSnapshot(
            task_id=self._task_id,
            tool_calls_started=self._tool_calls_started,
            active_tool_calls=self.active_tool_calls,
            total_tool_execution_seconds=self._total_tool_execution_seconds,
            elapsed_seconds=elapsed,
            remaining_tool_calls=remaining_calls,
            remaining_execution_seconds=remaining_execution,
            remaining_tool_execution_seconds=remaining_tool_execution,
            exhausted=exhausted,
        )

    def _elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)
