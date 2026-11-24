"""Exception hierarchy for the autonomous agent tool system."""


class ToolError(Exception):
    """Base exception for tool system errors."""


class ToolRegistrationError(ToolError):
    """Raised when a tool cannot be registered."""


class ToolAlreadyRegisteredError(ToolRegistrationError):
    """Raised when a tool name is already registered."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist in the registry."""


class ToolArgumentValidationError(ToolError):
    """Raised when tool arguments do not match their parameter schema."""