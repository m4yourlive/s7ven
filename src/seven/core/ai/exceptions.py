"""Exceptions for the S7VEN AI engine."""


class AIError(Exception):
    """Base exception for AI-related errors."""


class AIProviderError(AIError):
    """Raised when an AI provider encounters an error."""


class AIConfigurationError(AIError):
    """Raised when AI configuration is invalid or incomplete."""


class AIRequestError(AIError):
    """Raised when an AI generation request is invalid."""


class AIResponseError(AIError):
    """Raised when an AI provider returns an invalid response."""


class AIProviderUnavailableError(AIProviderError):
    """Raised when an AI provider is unavailable."""


class AIProviderTimeoutError(AIProviderError):
    """Raised when an AI provider request times out."""


class AIAuthenticationError(AIProviderError):
    """Raised when provider authentication fails."""


class AIRateLimitError(AIProviderError):
    """Raised when an AI provider rate limit is exceeded."""
