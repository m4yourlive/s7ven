
from seven.core.integrations.applications.interfaces import (
    ApplicationDiscovery,
    ApplicationLauncher,
)
from seven.core.integrations.applications.models import Application
from seven.core.integrations.applications.policy import (
    ApplicationAccessPolicy,
)


class ApplicationService:
    """Coordinate application discovery and controlled launching."""

    def __init__(
        self,
        discovery: ApplicationDiscovery,
        launcher: ApplicationLauncher,
        access_policy: ApplicationAccessPolicy,
    ) -> None:
        self._discovery = discovery
        self._launcher = launcher
        self._access_policy = access_policy
        self._is_running = False

    @property
    def name(self) -> str:
        return "applications"

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    def _ensure_running(self) -> None:
        if not self._is_running:
            raise RuntimeError(
                "Application service is not running."
            )

    def discover(self) -> tuple[Application, ...]:
        """Discover available applications."""

        self._ensure_running()
        return self._discovery.discover()

    def launch(self, application: Application) -> int:
        """Authorize and launch an application."""

        self._ensure_running()

        if not isinstance(application, Application):
            raise TypeError(
                "application must be an Application instance."
            )

        self._access_policy.authorize(application)

        return self._launcher.launch(application)
