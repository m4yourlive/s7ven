class ConfigurationError(Exception):
    """Base exception for configuration errors."""

class ConfigurationFileNotFoundError(ConfigurationError):
    """Raised when a configuration file does not exist."""

class ConfigurationLoadError(ConfigurationError):
    """Raised when configuration cannot be loaded."""

class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration fails validation."""