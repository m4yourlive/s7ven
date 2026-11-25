
"""Exceptions for workflow automation."""


class WorkflowError(Exception):
    """Base exception for workflow operations."""


class WorkflowValidationError(WorkflowError):
    """Raised when a workflow definition is invalid."""


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when a workflow cannot be found."""


class WorkflowAlreadyRunningError(WorkflowError):
    """Raised when a workflow execution is already active."""


class WorkflowCancellationError(WorkflowError):
    """Raised when workflow cancellation fails."""


class WorkflowBudgetExceededError(WorkflowExecutionError):
    """Raised when a workflow exceeds its execution budget."""
