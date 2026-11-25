
from dataclasses import dataclass
from threading import RLock

from seven.core.integrations.system.environment import (
    EnvironmentInformation,
    SystemInformation,
    detect_environment_information,
    detect_system_information,
)
from seven.core.integrations.system.platform import (
    PlatformInfo,
    detect_platform,
)
from seven.core.integrations.system.resources import (
    SystemResourceSnapshot,
    detect_system_resource_snapshot,
)


@dataclass(frozen=True, slots=True)
class SystemIntegrationSnapshot:
    platform: PlatformInfo
    system: SystemInformation
    environment: EnvironmentInformation
    resources: SystemResourceSnapshot


class SystemInformationService:
    """Provide consolidated system and environment information."""

    def __init__(self, cpu_interval: float = 0.1) -> None:
        if cpu_interval < 0:
            raise ValueError("CPU sampling interval cannot be negative.")

        self._cpu_interval = cpu_interval
        self._running = False
        self._snapshot: SystemIntegrationSnapshot | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return "system_information"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def snapshot(self) -> SystemIntegrationSnapshot | None:
        with self._lock:
            return self._snapshot

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

            snapshot = self._collect_snapshot()

            self._snapshot = snapshot
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._snapshot = None

    def refresh(self) -> SystemIntegrationSnapshot:
        with self._lock:
            if not self._running:
                raise RuntimeError(
                    "System information service is not running."
                )

            snapshot = self._collect_snapshot()
            self._snapshot = snapshot

            return snapshot

    def _collect_snapshot(self) -> SystemIntegrationSnapshot:
        return SystemIntegrationSnapshot(
            platform=detect_platform(),
            system=detect_system_information(),
            environment=detect_environment_information(),
            resources=detect_system_resource_snapshot(
                cpu_interval=self._cpu_interval,
            ),
        )
