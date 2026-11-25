
from typing import Protocol, runtime_checkable


@runtime_checkable
class IntegrationService(Protocol):
    """
    Contract for all S7VEN system integration services.

    Every integration service must expose its identity,
    runtime status, and lifecycle operations.
    """

    @property
    def name(self) -> str:
        """Return the unique service name."""
        ...

    @property
    def is_running(self) -> bool:
        """Return whether the service is currently running."""
        ...

    def start(self) -> None:
        """Initialize and start the service."""
        ...

    def stop(self) -> None:
        """Stop the service and release its resources."""
        ...
