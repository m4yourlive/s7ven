
import pytest

from seven.core.integrations import IntegrationService
from seven.core.integrations.system import (
    SystemInformationService,
    SystemIntegrationSnapshot,
)


def test_system_service_implements_integration_contract() -> None:
    service = SystemInformationService(cpu_interval=0)

    assert isinstance(service, IntegrationService)
    assert service.name == "system_information"
    assert service.is_running is False


def test_system_service_starts_and_collects_snapshot() -> None:
    service = SystemInformationService(cpu_interval=0)

    service.start()

    assert service.is_running is True
    assert isinstance(service.snapshot, SystemIntegrationSnapshot)
    assert service.snapshot.platform.system
    assert service.snapshot.system.hostname
    assert service.snapshot.resources.memory.total_bytes > 0


def test_system_service_start_is_idempotent() -> None:
    service = SystemInformationService(cpu_interval=0)

    service.start()
    original_snapshot = service.snapshot

    service.start()

    assert service.snapshot is original_snapshot


def test_system_service_refreshes_snapshot() -> None:
    service = SystemInformationService(cpu_interval=0)

    service.start()
    result = service.refresh()

    assert isinstance(result, SystemIntegrationSnapshot)
    assert service.snapshot is result


def test_system_service_cannot_refresh_before_start() -> None:
    service = SystemInformationService(cpu_interval=0)

    with pytest.raises(RuntimeError):
        service.refresh()


def test_system_service_stops_and_clears_snapshot() -> None:
    service = SystemInformationService(cpu_interval=0)

    service.start()
    service.stop()

    assert service.is_running is False
    assert service.snapshot is None


def test_system_service_stop_is_idempotent() -> None:
    service = SystemInformationService(cpu_interval=0)

    service.start()
    service.stop()
    service.stop()

    assert service.is_running is False
    assert service.snapshot is None


def test_system_service_rejects_negative_cpu_interval() -> None:
    with pytest.raises(ValueError):
        SystemInformationService(cpu_interval=-1)
