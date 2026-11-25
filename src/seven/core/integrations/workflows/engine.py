
"""Workflow execution engine with dependency, cancellation, and budget handling."""

import asyncio
import inspect
import time
from typing import Any, Protocol

from seven.core.integrations.workflows.exceptions import (
    WorkflowBudgetExceededError,
    WorkflowExecutionError,
)
from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepResult,
    WorkflowStepStatus,
)
from seven.core.integrations.workflows.policy import WorkflowExecutionPolicy


class WorkflowActionHandler(Protocol):
    """Handle workflow action execution."""

    async def execute(self, action: str, parameters: dict[str, Any]) -> Any:
        """Execute an action and return its output."""
        ...


class WorkflowEngine:
    """Execute workflows while enforcing safety and execution limits."""

    def __init__(
        self,
        action_handler: WorkflowActionHandler,
        validator: Any,
        execution_store: Any | None = None,
        policy: WorkflowExecutionPolicy | None = None,
    ) -> None:
        if action_handler is None:
            raise ValueError("Action handler is required.")

        if validator is None:
            raise ValueError("Workflow validator is required.")

        self._action_handler = action_handler
        self._validator = validator
        self._execution_store = execution_store
        self._policy = policy or WorkflowExecutionPolicy()

    async def execute(
        self,
        workflow: Workflow,
        cancellation_event: asyncio.Event | None = None,
    ) -> WorkflowExecution:
        """Validate and execute a workflow."""

        validation_result = self._validator.validate(workflow)

        if inspect.isawaitable(validation_result):
            await validation_result

        started_at = time.monotonic()
        results: list[WorkflowStepResult] = []
        pending_steps = list(workflow.steps)

        try:
            self._validate_budget_definition(workflow)

            execution = WorkflowExecution(
                workflow_id=workflow.workflow_id,
                status=WorkflowStatus.RUNNING,
                step_results=(),
            )

            while pending_steps:
                if self._is_cancelled(cancellation_event):
                    self._mark_skipped(pending_steps, results)
                    return await self._finish(
                        execution,
                        WorkflowStatus.CANCELLED,
                        results,
                    )

                self._check_duration(started_at)

                completed_ids = {
                    result.step_id
                    for result in results
                    if result.status == WorkflowStepStatus.COMPLETED
                }

                ready_steps = [
                    step
                    for step in pending_steps
                    if all(
                        dependency in completed_ids
                        for dependency in step.dependencies
                    )
                ]

                if not ready_steps:
                    raise WorkflowExecutionError(
                        "Workflow cannot progress because dependencies are unresolved."
                    )

                step = ready_steps[0]
                pending_steps.remove(step)

                if self._is_cancelled(cancellation_event):
                    pending_steps.insert(0, step)
                    self._mark_skipped(pending_steps, results)
                    return await self._finish(
                        execution,
                        WorkflowStatus.CANCELLED,
                        results,
                    )

                self._check_duration(started_at)

                result = await self._execute_step(step)
                results.append(result)

                if result.status == WorkflowStepStatus.FAILED:
                    self._mark_skipped(pending_steps, results)
                    return await self._finish(
                        execution,
                        WorkflowStatus.FAILED,
                        results,
                        error=result.error,
                    )

                if self._is_cancelled(cancellation_event):
                    self._mark_skipped(pending_steps, results)
                    return await self._finish(
                        execution,
                        WorkflowStatus.CANCELLED,
                        results,
                    )

                self._check_duration(started_at)

            return await self._finish(
                execution,
                WorkflowStatus.COMPLETED,
                results,
            )

        except asyncio.CancelledError:
            self._mark_skipped(pending_steps, results)
            await self._save_cancelled(workflow, results)
            raise

        except WorkflowExecutionError as error:
            self._mark_skipped(pending_steps, results)
            return await self._finish(
                WorkflowExecution(
                    workflow_id=workflow.workflow_id,
                    status=WorkflowStatus.RUNNING,
                    step_results=(),
                ),
                WorkflowStatus.FAILED,
                results,
                error=str(error),
            )

        except Exception as error:
            self._mark_skipped(pending_steps, results)
            execution = WorkflowExecution(
                workflow_id=workflow.workflow_id,
                status=WorkflowStatus.RUNNING,
                step_results=(),
            )
            return await self._finish(
                execution,
                WorkflowStatus.FAILED,
                results,
                error=str(error),
            )

    def _validate_budget_definition(self, workflow: Workflow) -> None:
        """Validate workflow size and action permissions before execution."""

        if len(workflow.steps) > self._policy.max_steps:
            raise WorkflowBudgetExceededError(
                f"Workflow contains {len(workflow.steps)} steps; "
                f"the maximum allowed is {self._policy.max_steps}."
            )

        allowed_actions = self._policy.allowed_actions

        if allowed_actions is not None:
            unauthorized = [
                step.action
                for step in workflow.steps
                if step.action not in allowed_actions
            ]

            if unauthorized:
                actions = ", ".join(sorted(set(unauthorized)))
                raise WorkflowExecutionError(
                    f"Workflow contains unauthorized actions: {actions}."
                )

    def _check_duration(self, started_at: float) -> None:
        """Raise when the workflow duration budget has been exhausted."""

        elapsed = time.monotonic() - started_at

        if elapsed >= self._policy.max_duration_seconds:
            raise WorkflowBudgetExceededError(
                "Workflow exceeded its maximum execution duration."
            )

    async def _execute_step(self, step: WorkflowStep) -> WorkflowStepResult:
        """Execute one workflow step and capture its result."""

        try:
            output = self._action_handler.execute(
                step.action,
                step.parameters,
            )

            if inspect.isawaitable(output):
                output = await output

            return WorkflowStepResult(
                step_id=step.step_id,
                status=WorkflowStepStatus.COMPLETED,
                output=output,
            )

        except asyncio.CancelledError:
            raise

        except Exception as error:
            return WorkflowStepResult(
                step_id=step.step_id,
                status=WorkflowStepStatus.FAILED,
                error=str(error),
            )

    def _mark_skipped(
        self,
        steps: list[WorkflowStep],
        results: list[WorkflowStepResult],
    ) -> None:
        """Mark all pending steps as skipped."""

        existing_ids = {result.step_id for result in results}

        for step in steps:
            if step.step_id in existing_ids:
                continue

            results.append(
                WorkflowStepResult(
                    step_id=step.step_id,
                    status=WorkflowStepStatus.SKIPPED,
                )
            )

    def _is_cancelled(
        self,
        cancellation_event: asyncio.Event | None,
    ) -> bool:
        """Return whether cooperative cancellation was requested."""

        return (
            cancellation_event is not None
            and cancellation_event.is_set()
        )

    async def _finish(
        self,
        execution: WorkflowExecution,
        status: WorkflowStatus,
        results: list[WorkflowStepResult],
        error: str | None = None,
    ) -> WorkflowExecution:
        """Build and persist the final execution result."""

        completed = WorkflowExecution(
            workflow_id=execution.workflow_id,
            status=status,
            step_results=tuple(results),
            execution_id=execution.execution_id,
            error=error,
        )

        if self._execution_store is not None:
            saved = self._execution_store.save(completed)

            if inspect.isawaitable(saved):
                await saved

        return completed

    async def _save_cancelled(
        self,
        workflow: Workflow,
        results: list[WorkflowStepResult],
    ) -> None:
        """Persist execution state when the current task is cancelled."""

        execution = WorkflowExecution(
            workflow_id=workflow.workflow_id,
            status=WorkflowStatus.CANCELLED,
            step_results=tuple(results),
            error="Workflow task was cancelled.",
        )

        if self._execution_store is not None:
            saved = self._execution_store.save(execution)

            if inspect.isawaitable(saved):
                await saved
