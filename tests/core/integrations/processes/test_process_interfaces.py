
"""Tests for process operation contracts."""

from seven.core.integrations.processes.interfaces import (
    ProcessDiscovery,
    ProcessInspector,
    ProcessTerminator,
)
from seven.core.integrations.processes.models import ProcessInfo


class DiscoveryImplementation:
    def discover(self) -> tuple[ProcessInfo, ...]:
        return (ProcessInfo(pid=1234, name="python"),)


class InspectorImplementation:
    def inspect(self, pid: int) -> ProcessInfo:
        return ProcessInfo(pid=pid, name="python")


class TerminatorImplementation:
    def terminate(self, pid: int) -> None:
        return None


def test_process_discovery_protocol_accepts_valid_implementation():
    assert isinstance(DiscoveryImplementation(), ProcessDiscovery)


def test_process_inspector_protocol_accepts_valid_implementation():
    assert isinstance(InspectorImplementation(), ProcessInspector)


def test_process_terminator_protocol_accepts_valid_implementation():
    assert isinstance(TerminatorImplementation(), ProcessTerminator)


def test_invalid_object_does_not_implement_process_discovery():
    assert not isinstance(object(), ProcessDiscovery)


def test_invalid_object_does_not_implement_process_inspector():
    assert not isinstance(object(), ProcessInspector)


def test_invalid_object_does_not_implement_process_terminator():
    assert not isinstance(object(), ProcessTerminator)
