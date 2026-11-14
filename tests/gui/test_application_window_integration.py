"""Tests for WebView application and system tray integration."""

from typing import Any

import pytest

from seven.gui.application import WebViewApplication


class FakeWindow:
    """Fake WebView window for integration tests."""

    def __init__(self) -> None:
        self.restore_called = False
        self.show_called = False
        self.destroy_called = False

    def restore(self) -> None:
        self.restore_called = True

    def show(self) -> None:
        self.show_called = True

    def destroy(self) -> None:
        self.destroy_called = True


class FakeTrayController:
    """Fake tray controller for integration tests."""

    def __init__(
        self,
        *,
        on_show: Any,
        on_exit: Any,
        title: str,
    ) -> None:
        self.on_show = on_show
        self.on_exit = on_exit
        self.title = title


def test_show_window_restores_and_shows_window() -> None:
    application = WebViewApplication()
    window = FakeWindow()
    application._window = window

    result = application.show_window()

    assert result is True
    assert window.restore_called is True
    assert window.show_called is True


def test_show_window_returns_false_without_window() -> None:
    application = WebViewApplication()

    assert application.show_window() is False


def test_show_window_returns_false_after_close() -> None:
    application = WebViewApplication()
    window = FakeWindow()
    application._window = window
    application._closed = True

    assert application.show_window() is False
    assert window.show_called is False


def test_show_window_requires_show_method() -> None:
    class InvalidWindow:
        def restore(self) -> None:
            pass

    application = WebViewApplication()
    application._window = InvalidWindow()

    with pytest.raises(RuntimeError, match="does not support show"):
        application.show_window()


def test_create_tray_controller_connects_application_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seven.gui.tray as tray_module

    monkeypatch.setattr(
        tray_module,
        "TrayController",
        FakeTrayController,
    )

    application = WebViewApplication()
    tray = application.create_tray_controller()

    assert isinstance(tray, FakeTrayController)
    assert tray.on_show == application.show_window
    assert tray.on_exit == application.close
    assert tray.title == application.config.title


def test_create_tray_controller_reuses_existing_controller() -> None:
    existing_tray = object()
    application = WebViewApplication(tray_controller=existing_tray)

    result = application.create_tray_controller()

    assert result is existing_tray


def test_tray_open_callback_shows_application_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seven.gui.tray as tray_module

    monkeypatch.setattr(
        tray_module,
        "TrayController",
        FakeTrayController,
    )

    application = WebViewApplication()
    window = FakeWindow()
    application._window = window

    tray = application.create_tray_controller()
    tray.on_show()

    assert window.restore_called is True
    assert window.show_called is True


def test_tray_exit_callback_closes_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import seven.gui.tray as tray_module

    monkeypatch.setattr(
        tray_module,
        "TrayController",
        FakeTrayController,
    )

    application = WebViewApplication()
    window = FakeWindow()
    application._window = window

    tray = application.create_tray_controller()
    tray.on_exit()

    assert application.is_closed is True
    assert window.destroy_called is True
