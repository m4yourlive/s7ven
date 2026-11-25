
from pathlib import Path
from unittest.mock import Mock

import pytest

from seven.core.integrations.applications import (
    Application,
    ApplicationAccessPolicy,
    ApplicationService,
)


@pytest.fixture
def executable(tmp_path: Path) -> Path:
    target = tmp_path / "example.exe"
    target.write_text("", encoding="utf-8")
    return target


@pytest.fixture
def application(executable: Path) -> Application:
    return Application(
        name="Example",
        executable=executable,
    )


@pytest.fixture
def dependencies() -> tuple[Mock, Mock]:
    return Mock(), Mock()


@pytest.fixture
def service(
    dependencies: tuple[Mock, Mock],
    executable: Path,
) -> ApplicationService:
    discovery, launcher = dependencies

    return ApplicationService(
        discovery=discovery,
        launcher=launcher,
        access_policy=ApplicationAccessPolicy(
            allowed_executables=(executable,),
        ),
    )


def test_service_has_correct_identity(
    service: ApplicationService,
) -> None:
    assert service.name == "applications"


def test_service_starts_stopped(
    service: ApplicationService,
) -> None:
    assert service.is_running is False


def test_service_start_and_stop(
    service: ApplicationService,
) -> None:
    service.start()
    assert service.is_running is True

    service.stop()
    assert service.is_running is False


def test_service_start_is_idempotent(
    service: ApplicationService,
) -> None:
    service.start()
    service.start()

    assert service.is_running is True


def test_service_stop_is_idempotent(
    service: ApplicationService,
) -> None:
    service.stop()
    service.stop()

    assert service.is_running is False


def test_discover_requires_running_service(
    service: ApplicationService,
) -> None:
    with pytest.raises(RuntimeError):
        service.discover()


def test_launch_requires_running_service(
    service: ApplicationService,
    application: Application,
) -> None:
    with pytest.raises(RuntimeError):
        service.launch(application)


def test_discover_delegates_to_discovery(
    service: ApplicationService,
    dependencies: tuple[Mock, Mock],
    application: Application,
) -> None:
    discovery, _ = dependencies
    discovery.discover.return_value = (application,)

    service.start()

    assert service.discover() == (application,)
    discovery.discover.assert_called_once_with()


def test_launch_delegates_to_launcher(
    service: ApplicationService,
    dependencies: tuple[Mock, Mock],
    application: Application,
) -> None:
    _, launcher = dependencies
    launcher.launch.return_value = 1234

    service.start()

    assert service.launch(application) == 1234
    launcher.launch.assert_called_once_with(application)


def test_service_rejects_invalid_application(
    service: ApplicationService,
) -> None:
    service.start()

    with pytest.raises(TypeError):
        service.launch("example.exe")  # type: ignore[arg-type]


def test_service_rejects_unapproved_application(
    service: ApplicationService,
    dependencies: tuple[Mock, Mock],
    tmp_path: Path,
) -> None:
    _, launcher = dependencies
    service.start()

    unauthorized = Application(
        name="Unauthorized",
        executable=tmp_path / "unauthorized.exe",
    )

    with pytest.raises(PermissionError):
        service.launch(unauthorized)

    launcher.launch.assert_not_called()


def test_service_operations_rejected_after_stop(
    service: ApplicationService,
    application: Application,
) -> None:
    service.start()
    service.stop()

    with pytest.raises(RuntimeError):
        service.discover()

    with pytest.raises(RuntimeError):
        service.launch(application)
