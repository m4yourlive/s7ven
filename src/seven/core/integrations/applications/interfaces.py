
from typing import Protocol, runtime_checkable

from seven.core.integrations.applications.models import Application


@runtime_checkable
class ApplicationDiscovery(Protocol):
    """Contract for discovering available desktop applications."""

    def discover(self) -> tuple[Application, ...]:
        """Return discovered desktop applications."""
        ...


@runtime_checkable
class ApplicationLauncher(Protocol):
    """Contract for launching desktop applications."""

    def launch(self, application: Application) -> int:
        """Launch an application and return its process identifier."""
        ...
