
"""Execution budget models and resource limit definitions."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionBudget:
    """Define resource limits for one autonomous agent task."""

    max_tool_calls: int = 25
    max_execution_seconds: float = 120.0
    max_tool_execution_seconds: float = 30.0
    max_total_tool_execution_seconds: float = 90.0
    max_concurrent_tool_calls: int = 3

    def __post_init__(self) -> None:
        integer_fields = (
            "max_tool_calls",
            "max_concurrent_tool_calls",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer.")

            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero.")

        float_fields = (
            "max_execution_seconds",
            "max_tool_execution_seconds",
            "max_total_tool_execution_seconds",
        )

        for field_name in float_fields:
            value = getattr(self, field_name)

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be a number.")

            if value <= 0 or value == float("inf") or value != value:
                raise ValueError(
                    f"{field_name} must be a finite positive number."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionLease:
    """Identify one active tool execution."""

    lease_id: UUID
    tool_name: str
    started_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.lease_id, UUID):
            raise TypeError("lease_id must be a UUID.")

        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        if not isinstance(self.started_at, (int, float)):
            raise TypeError("started_at must be a number.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionBudgetSnapshot:
    """Represent the current resource usage of an agent task."""

    task_id: UUID | None
    tool_calls_started: int
    active_tool_calls: int
    total_tool_execution_seconds: float
    elapsed_seconds: float
    remaining_tool_calls: int
    remaining_execution_seconds: float
    remaining_tool_execution_seconds: float
    exhausted: bool

    def __post_init__(self) -> None:
        if self.task_id is not None and not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID or None.")

        integer_fields = (
            "tool_calls_started",
            "active_tool_calls",
            "remaining_tool_calls",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer.")

            if value < 0:
                raise ValueError(f"{field_name} cannot be negative.")

        float_fields = (
            "total_tool_execution_seconds",
            "elapsed_seconds",
            "remaining_execution_seconds",
            "remaining_tool_execution_seconds",
        )

        for field_name in float_fields:
            value = getattr(self, field_name)

            if not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number.")

            if value < 0:
                raise ValueError(f"{field_name} cannot be negative.")

        if not isinstance(self.exhausted, bool):
            raise TypeError("exhausted must be a boolean.")


class ExecutionBudgetExceededError(RuntimeError):
    """Raised when an execution exceeds its configured resource budget."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason cannot be empty.")

        self.reason = reason
        super().__init__(reason)
