
"""Tests for workflow contracts."""

from uuid import UUID, uuid4

from seven.core.integrations.workflows.interfaces import (
    WorkflowExecutionStore,
    WorkflowExecutor,
    WorkflowValidator,
)
from seven.core.integrations.workflows.models import (
    Workflow,
    WorkflowExecution,
)


class ExecutorImplementation:
    async def execute(
        self,
        workflow: Workflow,
    ) -> WorkflowExecution:
        return WorkflowExecution(workflow_id=workflow.workflow_id)


class ValidatorImplementation:
    def validate(self, workflow: Workflow) -> None:
        return None


class ExecutionStoreImplementation:
    async def save(self, execution: WorkflowExecution) -> None:
        return None

    async def get(
        self,
        execution_id: UUID,
    ) -> WorkflowExecution | None:
        return None


def test_executor_protocol_accepts_valid_implementation():
    assert isinstance(ExecutorImplementation(), WorkflowExecutor)


def test_validator_protocol_accepts_valid_implementation():
    assert isinstance(ValidatorImplementation(), WorkflowValidator)


def test_execution_store_protocol_accepts_valid_implementation():
    assert isinstance(
        ExecutionStoreImplementation(),
        WorkflowExecutionStore,
    )


def test_invalid_object_does_not_implement_executor():
    assert not isinstance(object(), WorkflowExecutor)


def test_invalid_object_does_not_implement_validator():
    assert not isinstance(object(), WorkflowValidator)


def test_invalid_object_does_not_implement_execution_store():
    assert not isinstance(object(), WorkflowExecutionStore)


def test_execution_store_accepts_uuid_identifier():
    implementation = ExecutionStoreImplementation()

    assert implementation.get.__annotations__["execution_id"] is UUID
    assert isinstance(uuid4(), UUID)
