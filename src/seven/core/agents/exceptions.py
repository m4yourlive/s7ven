"""Exception hierarchy for the S7VEN autonomous agent."""


class AgentError(Exception):
    """Base exception for all autonomous agent errors."""


class AgentConfigurationError(AgentError):
    """Raised when autonomous agent configuration is invalid."""


class AgentTaskError(AgentError):
    """Raised when an agent task cannot be processed."""


class AgentPlanningError(AgentTaskError):
    """Raised when an agent cannot create or validate a task plan."""


class AgentExecutionError(AgentTaskError):
    """Raised when an agent step fails during execution."""


class AgentStateError(AgentError):
    """Raised when an agent lifecycle state operation is invalid."""


class AgentCancelledError(AgentTaskError):
    """Raised when an agent task is cancelled."""