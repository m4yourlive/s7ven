
from seven.core.integrations import IntegrationService


class FakeIntegrationService:
    def __init__(self) -> None:
        self._running = False

    @property
    def name(self) -> str:
        return "FakeIntegrationService"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False


def test_service_implements_integration_contract() -> None:
    service = FakeIntegrationService()

    assert isinstance(service, IntegrationService)
    assert service.name == "FakeIntegrationService"
    assert service.is_running is False


def test_service_lifecycle_contract() -> None:
    service = FakeIntegrationService()

    service.start()

    assert service.is_running is True

    service.stop()

    assert service.is_running is False


def test_invalid_object_does_not_implement_contract() -> None:
    class InvalidService:
        pass

    assert not isinstance(InvalidService(), IntegrationService)
