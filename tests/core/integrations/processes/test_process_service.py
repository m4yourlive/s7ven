
"""Tests for process integration service."""

import pytest

from seven.core.integrations.processes.models import ProcessInfo
from seven.core.integrations.processes.policy import ProcessAccessPolicy
from seven.core.integrations.processes.service import ProcessService


class FakeDiscovery:
    def __init__(self):
        self.called = False

    def discover(self):
        self.called = True
        return (ProcessInfo(pid=1234, name="python"),)


class FakeInspector:
    def __init__(self):
        self.received_pid = None

    def inspect(self, pid):
        self.received_pid = pid
        return ProcessInfo(pid=pid, name="python")


class FakeTerminator:
    def __init__(self):
        self.received_pid = None

    def terminate(self, pid):
        self.received_pid = pid


def create_service(
    *,
    access_policy=None,
):
    discovery = FakeDiscovery()
    inspector = FakeInspector()
    terminator = FakeTerminator()

    service = ProcessService(
        discovery=discovery,
        inspector=inspector,
        terminator=terminator,
        access_policy=access_policy,
    )

    return service, discovery, inspector, terminator


def test_service_has_expected_name():
    service, *_ = create_service()

    assert service.name == "processes"


def test_service_is_stopped_by_default():
    service, *_ = create_service()

    assert service.is_running is False


def test_service_start_changes_running_state():
    service, *_ = create_service()

    service.start()

    assert service.is_running is True


def test_service_start_is_idempotent():
    service, *_ = create_service()

    service.start()
    service.start()

    assert service.is_running is True


def test_service_stop_changes_running_state():
    service, *_ = create_service()

    service.start()
    service.stop()

    assert service.is_running is False


def test_service_stop_is_idempotent():
    service, *_ = create_service()

    service.stop()
    service.stop()

    assert service.is_running is False


def test_discovery_delegates_to_discovery_implementation():
    service, discovery, *_ = create_service()
    service.start()

    result = service.discover()

    assert discovery.called is True
    assert len(result) == 1
    assert result[0].pid == 1234


def test_inspection_delegates_to_inspector():
    service, _, inspector, _ = create_service()
    service.start()

    result = service.inspect(1234)

    assert inspector.received_pid == 1234
    assert result.pid == 1234


def test_termination_delegates_to_terminator_when_authorized():
    policy = ProcessAccessPolicy(allow_termination=True)
    service, _, _, terminator = create_service(
        access_policy=policy,
    )
    service.start()

    service.terminate(1234)

    assert terminator.received_pid == 1234


def test_termination_is_denied_by_default():
    service, _, _, terminator = create_service()
    service.start()

    with pytest.raises(PermissionError):
        service.terminate(1234)

    assert terminator.received_pid is None


def test_discovery_requires_running_service():
    service, discovery, *_ = create_service()

    with pytest.raises(RuntimeError):
        service.discover()

    assert discovery.called is False


def test_inspection_requires_running_service():
    service, _, inspector, _ = create_service()

    with pytest.raises(RuntimeError):
        service.inspect(1234)

    assert inspector.received_pid is None


def test_termination_requires_running_service():
    policy = ProcessAccessPolicy(allow_termination=True)
    service, _, _, terminator = create_service(
        access_policy=policy,
    )

    with pytest.raises(RuntimeError):
        service.terminate(1234)

    assert terminator.received_pid is None


def test_discovery_respects_access_policy():
    policy = ProcessAccessPolicy(allow_discovery=False)
    service, discovery, *_ = create_service(
        access_policy=policy,
    )
    service.start()

    with pytest.raises(PermissionError):
        service.discover()

    assert discovery.called is False


def test_inspection_respects_access_policy():
    policy = ProcessAccessPolicy(allow_inspection=False)
    service, _, inspector, _ = create_service(
        access_policy=policy,
    )
    service.start()

    with pytest.raises(PermissionError):
        service.inspect(1234)

    assert inspector.received_pid is None


def test_termination_respects_protected_process_policy():
    policy = ProcessAccessPolicy(
        allow_termination=True,
        protected_pids=frozenset({1234}),
    )
    service, _, _, terminator = create_service(
        access_policy=policy,
    )
    service.start()

    with pytest.raises(PermissionError):
        service.terminate(1234)

    assert terminator.received_pid is None


def test_service_can_restart_after_stop():
    service, discovery, *_ = create_service()

    service.start()
    service.stop()
    service.start()

    result = service.discover()

    assert service.is_running is True
    assert discovery.called is True
    assert len(result) == 1
