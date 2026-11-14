from __future__ import annotations

import importlib
from typing import Any

import pytest

main_module = importlib.import_module("seven.app.main")


class FakeRuntimeManager:
    """Fake runtime manager for application integration tests."""


class FakeContainer:
    """Fake dependency injection container."""

    def __init__(self) -> None:
        self.runtime = FakeRuntimeManager()
        self.runtime_manager_calls = 0

    def runtime_manager(self) -> FakeRuntimeManager:
        self.runtime_manager_calls += 1
        return self.runtime


class FakeWebViewApplication:
    """Fake WebView application."""

    def __init__(
        self,
        *,
        runtime_manager: Any,
    ) -> None:
        self.runtime_manager = runtime_manager
        self.run_calls = 0

    def run(self) -> None:
        self.run_calls += 1


def test_main_creates_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_containers: list[FakeContainer] = []

    def container_factory() -> FakeContainer:
        container = FakeContainer()
        created_containers.append(container)
        return container

    monkeypatch.setattr(
        main_module,
        "SevenContainer",
        container_factory,
    )

    monkeypatch.setattr(
        main_module,
        "WebViewApplication",
        FakeWebViewApplication,
    )

    result = main_module.main()

    assert result == 0
    assert len(created_containers) == 1


def test_main_passes_runtime_manager_to_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = FakeContainer()
    applications: list[FakeWebViewApplication] = []

    def application_factory(
        *,
        runtime_manager: Any,
    ) -> FakeWebViewApplication:
        application = FakeWebViewApplication(
            runtime_manager=runtime_manager,
        )
        applications.append(application)
        return application

    monkeypatch.setattr(
        main_module,
        "SevenContainer",
        lambda: container,
    )

    monkeypatch.setattr(
        main_module,
        "WebViewApplication",
        application_factory,
    )

    main_module.main()

    assert container.runtime_manager_calls == 1
    assert len(applications) == 1
    assert applications[0].runtime_manager is container.runtime


def test_main_runs_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applications: list[FakeWebViewApplication] = []

    def application_factory(
        *,
        runtime_manager: Any,
    ) -> FakeWebViewApplication:
        application = FakeWebViewApplication(
            runtime_manager=runtime_manager,
        )
        applications.append(application)
        return application

    monkeypatch.setattr(
        main_module,
        "SevenContainer",
        FakeContainer,
    )

    monkeypatch.setattr(
        main_module,
        "WebViewApplication",
        application_factory,
    )

    result = main_module.main()

    assert result == 0
    assert len(applications) == 1
    assert applications[0].run_calls == 1
