
from seven.core.integrations import IntegrationServiceManager
from seven.core.integrations.system import SystemInformationService


def test_system_service_can_be_registered() -> None:
    service = SystemInformationService(cpu_interval=0)
    manager = IntegrationServiceManager(services=[service])

    assert manager.services == (service,)


def test_system_service_starts_through_integration_manager() -> None:
    service = SystemInformationService(cpu_interval=0)
    manager = IntegrationServiceManager(services=[service])

    manager.start()

    assert manager.is_running is True
    assert service.is_running is True
    assert service.snapshot is not None


def test_system_service_stops_through_integration_manager() -> None:
    service = SystemInformationService(cpu_interval=0)
    manager = IntegrationServiceManager(services=[service])

    manager.start()
    manager.stop()

    assert manager.is_running is False
    assert service.is_running is False
    assert service.snapshot is None


def test_system_service_registration_is_idempotent() -> None:
    service = SystemInformationService(cpu_interval=0)
    manager = IntegrationServiceManager(services=[service])

    manager.start()
    manager.start()

    assert service.is_running is True
    assert len(manager.active_services) == 1
