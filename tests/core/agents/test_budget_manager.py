
"""Tests for autonomous agent execution budgets."""

import math
import time
from uuid import uuid4

import pytest

from seven.core.agents.budget import (
    ExecutionBudget,
    ExecutionBudgetExceededError,
    ExecutionBudgetSnapshot,
    ExecutionLease,
)
from seven.core.agents.budget_manager import ExecutionBudgetManager


class TestExecutionBudget:
    def test_default_budget(self) -> None:
        budget = ExecutionBudget()

        assert budget.max_tool_calls == 25
        assert budget.max_execution_seconds == 120.0
        assert budget.max_tool_execution_seconds == 30.0
        assert budget.max_total_tool_execution_seconds == 90.0
        assert budget.max_concurrent_tool_calls == 3

    @pytest.mark.parametrize(
        "field_name",
        [
            "max_tool_calls",
            "max_concurrent_tool_calls",
        ],
    )
    def test_integer_limits_must_be_positive(
        self,
        field_name: str,
    ) -> None:
        with pytest.raises(ValueError):
            ExecutionBudget(**{field_name: 0})

    @pytest.mark.parametrize(
        "field_name",
        [
            "max_execution_seconds",
            "max_tool_execution_seconds",
            "max_total_tool_execution_seconds",
        ],
    )
    def test_time_limits_must_be_positive(
        self,
        field_name: str,
    ) -> None:
        with pytest.raises(ValueError):
            ExecutionBudget(**{field_name: 0})

    def test_rejects_boolean_integer_limit(self) -> None:
        with pytest.raises(TypeError):
            ExecutionBudget(max_tool_calls=True)

    def test_rejects_invalid_integer_limit(self) -> None:
        with pytest.raises(TypeError):
            ExecutionBudget(max_tool_calls=1.5)

    @pytest.mark.parametrize(
        "value",
        [
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
    )
    def test_rejects_non_finite_time_limits(self, value: float) -> None:
        with pytest.raises(ValueError):
            ExecutionBudget(max_execution_seconds=value)


class TestExecutionLease:
    def test_lease_creation(self) -> None:
        lease = ExecutionLease(
            lease_id=uuid4(),
            tool_name="search",
            started_at=time.monotonic(),
        )

        assert lease.tool_name == "search"

    def test_rejects_invalid_lease_id(self) -> None:
        with pytest.raises(TypeError):
            ExecutionLease(
                lease_id="invalid",
                tool_name="search",
                started_at=0.0,
            )

    def test_rejects_empty_tool_name(self) -> None:
        with pytest.raises(ValueError):
            ExecutionLease(
                lease_id=uuid4(),
                tool_name=" ",
                started_at=0.0,
            )


class TestExecutionBudgetSnapshot:
    def test_snapshot_creation(self) -> None:
        snapshot = ExecutionBudgetSnapshot(
            task_id=None,
            tool_calls_started=1,
            active_tool_calls=0,
            total_tool_execution_seconds=1.0,
            elapsed_seconds=2.0,
            remaining_tool_calls=24,
            remaining_execution_seconds=118.0,
            remaining_tool_execution_seconds=89.0,
            exhausted=False,
        )

        assert snapshot.tool_calls_started == 1
        assert snapshot.exhausted is False

    def test_rejects_negative_tool_calls(self) -> None:
        with pytest.raises(ValueError):
            ExecutionBudgetSnapshot(
                task_id=None,
                tool_calls_started=-1,
                active_tool_calls=0,
                total_tool_execution_seconds=0.0,
                elapsed_seconds=0.0,
                remaining_tool_calls=25,
                remaining_execution_seconds=120.0,
                remaining_tool_execution_seconds=90.0,
                exhausted=False,
            )


class TestExecutionBudgetManager:
    def test_manager_creation(self) -> None:
        task_id = uuid4()
        manager = ExecutionBudgetManager(task_id=task_id)

        assert manager.task_id == task_id
        assert manager.active_tool_calls == 0
        assert manager.tool_calls_started == 0

    def test_start_tool_call(self) -> None:
        manager = ExecutionBudgetManager()

        lease = manager.start_tool_call("search")

        assert lease.tool_name == "search"
        assert manager.active_tool_calls == 1
        assert manager.tool_calls_started == 1

    def test_complete_tool_call(self) -> None:
        manager = ExecutionBudgetManager()

        lease = manager.start_tool_call("search")
        duration = manager.complete_tool_call(lease)

        assert duration >= 0
        assert manager.active_tool_calls == 0

    def test_rejects_unknown_lease(self) -> None:
        manager = ExecutionBudgetManager()

        lease = ExecutionLease(
            lease_id=uuid4(),
            tool_name="search",
            started_at=time.monotonic(),
        )

        with pytest.raises(ValueError):
            manager.complete_tool_call(lease)

    def test_rejects_double_completion(self) -> None:
        manager = ExecutionBudgetManager()

        lease = manager.start_tool_call("search")
        manager.complete_tool_call(lease)

        with pytest.raises(ValueError):
            manager.complete_tool_call(lease)

    def test_enforces_max_tool_calls(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(max_tool_calls=1)
        )

        lease = manager.start_tool_call("first")
        manager.complete_tool_call(lease)

        with pytest.raises(ExecutionBudgetExceededError):
            manager.start_tool_call("second")

    def test_enforces_max_concurrent_tool_calls(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(max_concurrent_tool_calls=1)
        )

        manager.start_tool_call("first")

        with pytest.raises(ExecutionBudgetExceededError):
            manager.start_tool_call("second")

    def test_allows_new_call_after_active_call_completes(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(
                max_tool_calls=3,
                max_concurrent_tool_calls=1,
            )
        )

        first = manager.start_tool_call("first")
        manager.complete_tool_call(first)

        second = manager.start_tool_call("second")

        assert manager.active_tool_calls == 1
        assert manager.tool_calls_started == 2

        manager.complete_tool_call(second)

    def test_enforces_task_execution_timeout(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(max_execution_seconds=0.01)
        )

        time.sleep(0.02)

        with pytest.raises(ExecutionBudgetExceededError):
            manager.start_tool_call("search")

    def test_enforces_individual_tool_execution_timeout(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(
                max_tool_execution_seconds=0.01,
                max_total_tool_execution_seconds=10.0,
            )
        )

        lease = manager.start_tool_call("slow_tool")
        time.sleep(0.02)

        with pytest.raises(ExecutionBudgetExceededError):
            manager.complete_tool_call(lease)

        assert manager.active_tool_calls == 0

    def test_enforces_total_tool_execution_time(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(
                max_tool_execution_seconds=10.0,
                max_total_tool_execution_seconds=0.01,
            )
        )

        lease = manager.start_tool_call("slow_tool")
        time.sleep(0.02)

        with pytest.raises(ExecutionBudgetExceededError):
            manager.complete_tool_call(lease)

        assert manager.active_tool_calls == 0

    def test_get_tool_timeout(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(
                max_execution_seconds=60.0,
                max_tool_execution_seconds=10.0,
                max_total_tool_execution_seconds=30.0,
            )
        )

        timeout = manager.get_tool_timeout()

        assert 0 < timeout <= 10.0

    def test_snapshot_tracks_usage(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(max_tool_calls=5)
        )

        lease = manager.start_tool_call("search")
        snapshot = manager.snapshot()

        assert snapshot.tool_calls_started == 1
        assert snapshot.active_tool_calls == 1
        assert snapshot.remaining_tool_calls == 4

        manager.complete_tool_call(lease)

    def test_snapshot_marks_concurrency_limit_as_exhausted(self) -> None:
        manager = ExecutionBudgetManager(
            budget=ExecutionBudget(max_concurrent_tool_calls=1)
        )

        manager.start_tool_call("search")

        assert manager.snapshot().exhausted is True

    def test_invalid_tool_name_is_rejected(self) -> None:
        manager = ExecutionBudgetManager()

        with pytest.raises(ValueError):
            manager.start_tool_call(" ")

    def test_invalid_budget_type_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            ExecutionBudgetManager(budget="invalid")

    def test_invalid_task_id_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            ExecutionBudgetManager(task_id="invalid")
