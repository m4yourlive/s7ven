
"""Domain models for workflow automation."""

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4


class WorkflowStatus(str, Enum):
    """Represent workflow execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepStatus(str, Enum):
    """Represent individual workflow step states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """Define an individual workflow action."""

    name: str
    action: str
    parameters: dict = field(default_factory=dict)
    step_id: UUID = field(default_factory=uuid4)
    dependencies: tuple[UUID, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Step name must be a string.")
        if not self.name.strip():
            raise ValueError("Step name cannot be empty.")

        if not isinstance(self.action, str):
            raise TypeError("Step action must be a string.")
        if not self.action.strip():
            raise ValueError("Step action cannot be empty.")

        if not isinstance(self.parameters, dict):
            raise TypeError("Step parameters must be a dictionary.")

        if not isinstance(self.step_id, UUID):
            raise TypeError("Step ID must be a UUID.")

        if not isinstance(self.dependencies, tuple):
            raise TypeError("Step dependencies must be a tuple.")

        if any(not isinstance(item, UUID) for item in self.dependencies):
            raise TypeError("Every dependency must be a UUID.")

        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("Step dependencies must be unique.")

        if self.step_id in self.dependencies:
            raise ValueError("A step cannot depend on itself.")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "action", self.action.strip())
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True, slots=True)
class Workflow:
    """Define a sequence of workflow steps."""

    name: str
    steps: tuple[WorkflowStep, ...]
    description: str | None = None
    workflow_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Workflow name must be a string.")
        if not self.name.strip():
            raise ValueError("Workflow name cannot be empty.")

        if not isinstance(self.steps, tuple):
            raise TypeError("Workflow steps must be a tuple.")

        if any(not isinstance(step, WorkflowStep) for step in self.steps):
            raise TypeError("Every workflow step must be a WorkflowStep.")

        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Workflow step IDs must be unique.")

        if self.description is not None and not isinstance(
            self.description, str
        ):
            raise TypeError("Workflow description must be a string.")

        if not isinstance(self.workflow_id, UUID):
            raise TypeError("Workflow ID must be a UUID.")

        object.__setattr__(self, "name", self.name.strip())


@dataclass(frozen=True, slots=True)
class WorkflowStepResult:
    """Store the result of an executed workflow step."""

    step_id: UUID
    status: WorkflowStepStatus
    output: object = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, UUID):
            raise TypeError("Step result ID must be a UUID.")
        if not isinstance(self.status, WorkflowStepStatus):
            raise TypeError("Step result status must be a WorkflowStepStatus.")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("Step result error must be a string or None.")


@dataclass(frozen=True, slots=True)
class WorkflowExecution:
    """Represent the result of a workflow execution."""

    workflow_id: UUID
    status: WorkflowStatus = WorkflowStatus.PENDING
    step_results: tuple[WorkflowStepResult, ...] = ()
    execution_id: UUID = field(default_factory=uuid4)
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workflow_id, UUID):
            raise TypeError("Workflow ID must be a UUID.")
        if not isinstance(self.status, WorkflowStatus):
            raise TypeError("Workflow status must be a WorkflowStatus.")
        if not isinstance(self.step_results, tuple):
            raise TypeError("Step results must be a tuple.")
        if any(
            not isinstance(result, WorkflowStepResult)
            for result in self.step_results
        ):
            raise TypeError("Every result must be a WorkflowStepResult.")
        if not isinstance(self.execution_id, UUID):
            raise TypeError("Execution ID must be a UUID.")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("Workflow error must be a string or None.")
