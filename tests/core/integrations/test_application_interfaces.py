
from pathlib import Path

from seven.core.integrations.applications import (
    Application,
    ApplicationDiscovery,
    ApplicationLauncher,
)


class FakeDiscovery:
    def discover(self) -> tuple[Application, ...]:
        return (
            Application(
                name="Example",
                executable=Path("example.exe"),
            ),
        )


class FakeLauncher:
    def launch(self, application: Application) -> int:
        return 1234


def test_fake_discovery_implements_protocol() -> None:
    discovery = FakeDiscovery()

    assert isinstance(discovery, ApplicationDiscovery)
    assert len(discovery.discover()) == 1


def test_fake_launcher_implements_protocol() -> None:
    launcher = FakeLauncher()
    application = Application(
        name="Example",
        executable=Path("example.exe"),
    )

    assert isinstance(launcher, ApplicationLauncher)
    assert launcher.launch(application) == 1234


def test_discovery_returns_application_instances() -> None:
    discovery = FakeDiscovery()

    applications = discovery.discover()

    assert all(
        isinstance(application, Application)
        for application in applications
    )
