
"""Exception hierarchy for the S7VEN AI engine."""


class AIError(Exception):
    """Base exception for all AI-related errors."""


class AIConfigurationError(AIError):
    """Raised when AI configuration is invalid or incomplete."""


class AIRequestError(AIError):
    """Raised when an AI generation request is invalid."""


class AIGenerationError(AIError):
    """Raised when an AI generation operation fails."""


class AIResponseError(AIError):
    """Raised when an AI provider returns an invalid response."""


class AIProviderError(AIError):
    """Base exception for AI provider-related failures."""


class AIProviderUnavailableError(AIProviderError):
    """Raised when an AI provider is unavailable."""


class AIProviderTimeoutError(AIProviderError):
    """Raised when an AI provider request times out."""


class AIProviderNetworkError(AIProviderError):
    """Raised when an AI provider encounters a network failure."""


class AIAuthenticationError(AIProviderError):
    """Raised when provider authentication fails."""


class AIRateLimitError(AIProviderError):
    """Raised when an AI provider rate limit is exceeded."""


class AIProviderCapabilityError(AIProviderError):
    """Raised when a provider cannot support a requested feature."""
