
"""Contracts for workflow automation."""

from typing import Protocol, runtime_checkable
from uuid import UUID

from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowExecution,
)


@runtime_checkable
class WorkflowExecutor(Protocol):
    """Define the contract for workflow execution."""

    async def execute(
        self,
        workflow: Workflow,
    ) -> WorkflowExecution:
        """Execute a workflow and return its result."""
        ...


@runtime_checkable
class WorkflowValidator(Protocol):
    """Define the contract for workflow validation."""

    def validate(self, workflow: Workflow) -> None:
        """Validate a workflow definition."""
        ...


@runtime_checkable
class WorkflowExecutionStore(Protocol):
    """Define the contract for workflow execution storage."""

    async def save(self, execution: WorkflowExecution) -> None:
        """Persist a workflow execution."""
        ...

    async def get(
        self,
        execution_id: UUID,
    ) -> WorkflowExecution | None:
        """Retrieve an execution by its identifier."""
        ...
