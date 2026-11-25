
"""Exceptions for the autonomous reasoning loop."""


class ReasoningError(Exception):
    """Base exception for reasoning loop failures."""


class ReasoningConfigurationError(ReasoningError):
    """Raised when reasoning configuration is invalid."""


class ReasoningExecutionError(ReasoningError):
    """Raised when reasoning execution fails."""


class ReasoningLimitExceededError(ReasoningError):
    """Raised when a reasoning execution limit is exceeded."""


class ReasoningTimeoutError(ReasoningError):
    """Raised when reasoning execution exceeds its timeout."""


class ReasoningToolError(ReasoningError):
    """Raised when tool execution fails during reasoning."""


class ReasoningProviderError(ReasoningError):
    """Raised when the AI provider fails during reasoning."""
