
"""Tests for the S7VEN system tray controller."""

from __future__ import annotations

import pytest

from seven.gui.tray import TrayController


class FakeTrayIcon:
    """Fake native tray icon."""

    def __init__(self) -> None:
        self.run_detached_calls = 0
        self.stop_calls = 0

    def run_detached(self) -> None:
        self.run_detached_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def create_controller(
    icon: FakeTrayIcon | None = None,
) -> tuple[TrayController, FakeTrayIcon]:
    fake_icon = icon or FakeTrayIcon()

    controller = TrayController(
        on_show=lambda: None,
        on_exit=lambda: None,
        icon_factory=lambda: fake_icon,
    )

    return controller, fake_icon


def test_tray_controller_can_be_initialized() -> None:
    controller, _ = create_controller()

    assert isinstance(controller, TrayController)
    assert controller.is_running is False


def test_tray_controller_rejects_invalid_show_callback() -> None:
    with pytest.raises(TypeError, match="on_show must be callable"):
        TrayController(
            on_show="invalid",  # type: ignore[arg-type]
            on_exit=lambda: None,
        )


def test_tray_controller_rejects_invalid_exit_callback() -> None:
    with pytest.raises(TypeError, match="on_exit must be callable"):
        TrayController(
            on_show=lambda: None,
            on_exit="invalid",  # type: ignore[arg-type]
        )


def test_tray_controller_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="Tray title cannot be empty"):
        TrayController(
            on_show=lambda: None,
            on_exit=lambda: None,
            title=" ",
        )


def test_start_creates_and_runs_tray_icon() -> None:
    controller, icon = create_controller()

    controller.start()

    assert controller.is_running is True
    assert icon.run_detached_calls == 1


def test_start_rejects_duplicate_start() -> None:
    controller, icon = create_controller()

    controller.start()

    with pytest.raises(RuntimeError, match="already running"):
        controller.start()

    assert icon.run_detached_calls == 1


def test_stop_stops_tray_icon() -> None:
    controller, icon = create_controller()

    controller.start()
    controller.stop()

    assert controller.is_running is False
    assert icon.stop_calls == 1


def test_stop_is_safe_when_not_running() -> None:
    controller, icon = create_controller()

    controller.stop()
    controller.stop()

    assert controller.is_running is False
    assert icon.stop_calls == 0


def test_tray_can_restart_after_stopping() -> None:
    icons: list[FakeTrayIcon] = []

    def icon_factory() -> FakeTrayIcon:
        icon = FakeTrayIcon()
        icons.append(icon)
        return icon

    controller = TrayController(
        on_show=lambda: None,
        on_exit=lambda: None,
        icon_factory=icon_factory,
    )

    controller.start()
    controller.stop()
    controller.start()

    assert len(icons) == 2
    assert icons[0].run_detached_calls == 1
    assert icons[0].stop_calls == 1
    assert icons[1].run_detached_calls == 1
    assert controller.is_running is True
