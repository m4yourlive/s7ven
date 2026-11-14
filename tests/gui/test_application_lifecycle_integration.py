"""Tests for application and system tray lifecycle integration."""

from __future__ import annotations

from typing import Any

from seven.gui.application import WebViewApplication


class FakeEvent:
    """Minimal fake pywebview event."""

    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def __iadd__(self, callback: Any) -> FakeEvent:
        self.callbacks.append(callback)
        return self

    def fire(self) -> Any:
        return self.callbacks[0]() if self.callbacks else None

class FakeWindowEvents:
    """Fake pywebview window events."""

    def __init__(self) -> None:
        self.closing = FakeEvent()

class FakeWindow:
    """Fake pywebview window."""

    def __init__(self) -> None:
        self.events = FakeWindowEvents()
        self.hide_called = False
        self.show_called = False
        self.restore_called = False
        self.destroy_called = False

    def hide(self) -> None:
        self.hide_called = True

    def show(self) -> None:
        self.show_called = True

    def restore(self) -> None:
        self.restore_called = True

    def destroy(self) -> None:
        self.destroy_called = True

class FakeTrayController:
    """Fake system tray controller."""

    def __init__(
        self,
        *,
        on_show: Any = None,
        on_exit: Any = None,
        title: str = "S7VEN",
    ) -> None:
        self.on_show = on_show
        self.on_exit = on_exit
        self.title = title
        self.is_running = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.is_running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_running = False

class FakeWebViewAPI:
    """Fake pywebview API."""

    def __init__(self, window: FakeWindow | None = None) -> None:
        self.window = window or FakeWindow()
        self.start_called = False

    def create_window(self, **_kwargs: Any) -> FakeWindow:
        return self.window

    def start(self, **_kwargs: Any) -> None:
        self.start_called = True

def create_application(
    *,
    tray: FakeTrayController | None = None,
) -> tuple[WebViewApplication, FakeWindow, FakeWebViewAPI]:
    """Create an application with fake platform dependencies."""
    window = FakeWindow()
    webview_api = FakeWebViewAPI(window)

    application = WebViewApplication(
        webview_api=webview_api,
        tray_controller=tray,
    )

    return application, window, webview_api

def test_window_closing_hides_window_when_tray_is_running() -> None:
    tray = FakeTrayController()
    tray.is_running = True

    application, window, _ = create_application(tray=tray)
    application.create_window(url="https://example.com")

    result = window.events.closing.fire()

    assert result is False
    assert window.hide_called is True
    assert application.is_closed is False

def test_window_closing_is_allowed_without_tray() -> None:
    application, window, _ = create_application()
    application.create_window(url="https://example.com")

    result = window.events.closing.fire()

    assert result is None
    assert window.hide_called is False

def test_window_closing_is_allowed_when_tray_is_stopped() -> None:
    tray = FakeTrayController()

    application, window, _ = create_application(tray=tray)
    application.create_window(url="https://example.com")

    result = window.events.closing.fire()

    assert result is None
    assert window.hide_called is False

def test_window_closing_is_allowed_after_exit_requested() -> None:
    tray = FakeTrayController()
    tray.is_running = True

    application, window, _ = create_application(tray=tray)
    application.create_window(url="https://example.com")
    application._exit_requested = True

    result = window.events.closing.fire()

    assert result is None
    assert window.hide_called is False

def test_close_stops_tray_and_destroys_window() -> None:
    tray = FakeTrayController()
    tray.is_running = True

    application, window, _ = create_application(tray=tray)
    application._window = window

    application.close()

    assert application.is_closed is True
    assert window.destroy_called is True
    assert tray.stop_calls == 1
    assert tray.is_running is False

def test_close_is_idempotent() -> None:
    tray = FakeTrayController()
    application, window, _ = create_application(tray=tray)
    application._window = window

    application.close()
    application.close()

    assert window.destroy_called is True
    assert tray.stop_calls == 1

def test_run_starts_tray_and_stops_it_after_webview_exits() -> None:
    tray = FakeTrayController()
    application, _, webview_api = create_application(tray=tray)

    application.run(url="https://example.com")

    assert webview_api.start_called is True
    assert tray.start_calls == 1
    assert tray.stop_calls == 1
    assert tray.is_running is False
    assert application.is_closed is True

def test_tray_exit_callback_performs_application_shutdown() -> None:
    tray = FakeTrayController()
    application, window, _ = create_application(tray=tray)
    application._window = window
    tray.on_exit = application.close
    tray.is_running = True

    tray.on_exit()

    assert application.is_closed is True
    assert window.destroy_called is True
    assert tray.is_running is False

def test_tray_open_callback_restores_hidden_window() -> None:
    tray = FakeTrayController()
    application, window, _ = create_application(tray=tray)
    application._window = window
    tray.on_show = application.show_window

    result = tray.on_show()

    assert result is True
    assert window.restore_called is True
    assert window.show_called is True