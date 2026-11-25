
"""Workflow definition validation."""

from seven.core.integrations.workflows.exceptions import (
    WorkflowValidationError,
)
from seven.core.integrations.workflows.models import Workflow


class DefaultWorkflowValidator:
    """Validate workflow structure and dependencies."""

    def validate(self, workflow: Workflow) -> None:
        """Validate workflow structure and step definitions."""
        if not isinstance(workflow, Workflow):
            raise WorkflowValidationError(
                "Workflow must be a Workflow instance."
            )

        if not workflow.steps:
            raise WorkflowValidationError(
                "Workflow must contain at least one step."
            )

        step_ids = set()

        for index, step in enumerate(workflow.steps):
            if not step.name.strip():
                raise WorkflowValidationError(
                    f"Step at index {index} has an empty name."
                )

            if not step.action.strip():
                raise WorkflowValidationError(
                    f"Step '{step.name}' has an empty action."
                )

            if step.step_id in step_ids:
                raise WorkflowValidationError(
                    f"Duplicate step ID found: {step.step_id}."
                )

            step_ids.add(step.step_id)

            if not isinstance(step.parameters, dict):
                raise WorkflowValidationError(
                    f"Step '{step.name}' parameters must be a dictionary."
                )

        known_ids = {step.step_id for step in workflow.steps}

        for step in workflow.steps:
            for dependency in step.dependencies:
                if dependency not in known_ids:
                    raise WorkflowValidationError(
                        f"Step '{step.name}' references an unknown dependency."
                    )

        self._validate_dependency_cycles(workflow)

    @staticmethod
    def _validate_dependency_cycles(workflow: Workflow) -> None:
        """Reject cyclic step dependencies."""
        dependencies = {
            step.step_id: set(step.dependencies)
            for step in workflow.steps
        }
        completed = set()

        while dependencies:
            ready = {
                step_id
                for step_id, required in dependencies.items()
                if required.issubset(completed)
            }

            if not ready:
                raise WorkflowValidationError(
                    "Workflow contains cyclic dependencies."
                )

            completed.update(ready)

            for step_id in ready:
                del dependencies[step_id]
