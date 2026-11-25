
"""Contracts for operating system process operations."""

from typing import Protocol, runtime_checkable

from seven.core.integrations.processes.models import ProcessInfo


@runtime_checkable
class ProcessDiscovery(Protocol):
    """Define the contract for discovering running processes."""

    def discover(self) -> tuple[ProcessInfo, ...]:
        """Return discovered running processes."""
        ...


@runtime_checkable
class ProcessInspector(Protocol):
    """Define the contract for inspecting a process."""

    def inspect(self, pid: int) -> ProcessInfo:
        """Return information about a process."""
        ...


@runtime_checkable
class ProcessTerminator(Protocol):
    """Define the contract for terminating a process."""

    def terminate(self, pid: int) -> None:
        """Request termination of a process."""
        ...
