
"""Lifecycle-managed process integration service."""

from seven.core.integrations.processes.discovery import (
    PsutilProcessDiscovery,
)
from seven.core.integrations.processes.inspection import (
    PsutilProcessInspector,
)
from seven.core.integrations.processes.interfaces import (
    ProcessDiscovery,
    ProcessInspector,
    ProcessTerminator,
)
from seven.core.integrations.processes.models import ProcessInfo
from seven.core.integrations.processes.policy import ProcessAccessPolicy
from seven.core.integrations.processes.termination import (
    PsutilProcessTerminator,
)


class ProcessService:
    """Coordinate process operations under an access policy."""

    def __init__(
        self,
        discovery: ProcessDiscovery | None = None,
        inspector: ProcessInspector | None = None,
        terminator: ProcessTerminator | None = None,
        access_policy: ProcessAccessPolicy | None = None,
    ) -> None:
        self._discovery = discovery or PsutilProcessDiscovery()
        self._inspector = inspector or PsutilProcessInspector()
        self._terminator = terminator or PsutilProcessTerminator()
        self._access_policy = access_policy or ProcessAccessPolicy()
        self._is_running = False

    @property
    def name(self) -> str:
        """Return the integration service name."""
        return "processes"

    @property
    def is_running(self) -> bool:
        """Return whether the service is running."""
        return self._is_running

    def start(self) -> None:
        """Start the process integration service."""
        if self._is_running:
            return

        self._is_running = True

    def stop(self) -> None:
        """Stop the process integration service."""
        if not self._is_running:
            return

        self._is_running = False

    def discover(self) -> tuple[ProcessInfo, ...]:
        """Discover running processes."""
        self._ensure_running()
        self._access_policy.authorize_discovery()
        return self._discovery.discover()

    def inspect(self, pid: int) -> ProcessInfo:
        """Inspect a process by its process identifier."""
        self._ensure_running()
        self._access_policy.authorize_inspection(pid)
        return self._inspector.inspect(pid)

    def terminate(self, pid: int) -> None:
        """Terminate a process when authorized."""
        self._ensure_running()
        self._access_policy.authorize_termination(pid)
        self._terminator.terminate(pid)

    def _ensure_running(self) -> None:
        """Ensure operations are only used while the service is running."""
        if not self._is_running:
            raise RuntimeError("Process service is not running.")
