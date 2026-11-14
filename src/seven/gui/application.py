"""S7VEN WebView application controller."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from seven.gui.bridge import FrontendBridge
from seven.gui.window import WindowConfig, create_webview_window

if TYPE_CHECKING:
    from seven.core.runtime.runtime_manager import RuntimeManager
    from seven.gui.tray import TrayController


class WebViewApplication:
    """Manage the WebView application and runtime lifecycle."""

    def __init__(
        self,
        config: WindowConfig | None = None,
        webview_api: Any | None = None,
        bridge: FrontendBridge | None = None,
        runtime_manager: RuntimeManager | None = None,
        tray_controller: TrayController | None = None,
    ) -> None:
        self.config = config or WindowConfig()
        self.bridge = bridge or FrontendBridge()
        self._webview_api = webview_api
        self._runtime_manager = runtime_manager
        self._tray_controller = tray_controller
        self._window: Any | None = None
        self._started = False
        self._closed = False
        self._runtime_started_by_app = False

    @property
    def window(self) -> Any | None:
        """Return the current WebView window."""
        return self._window

    @property
    def is_started(self) -> bool:
        """Return whether the application has started."""
        return self._started

    @property
    def is_closed(self) -> bool:
        """Return whether the application has closed."""
        return self._closed

    @property
    def owns_runtime(self) -> bool:
        """Return whether this application started the runtime."""
        return self._runtime_started_by_app

    @property
    def tray_controller(self) -> TrayController | None:
        """Return the configured system tray controller."""
        return self._tray_controller

    @staticmethod
    def _get_frontend_url() -> str:
        """Resolve the frontend entry point."""
        frontend_path = (
            Path(__file__).resolve().parent / "frontend" / "index.html"
        )

        if not frontend_path.is_file():
            raise FileNotFoundError(
                f"Frontend entry point not found: {frontend_path}"
            )

        return frontend_path.as_uri()

    def _get_webview_api(self) -> Any:
        """Load pywebview only when required."""
        if self._webview_api is None:
            import webview

            self._webview_api = webview

        return self._webview_api

    def create_window(
        self,
        url: str | None = None,
        html: str | None = None,
    ) -> Any:
        """Create and store the application window."""
        if self._closed:
            raise RuntimeError("Cannot create a window after application close.")

        if self._window is not None:
            raise RuntimeError("Application window has already been created.")

        if url is None and html is None:
            url = self._get_frontend_url()

        self._window = create_webview_window(
            webview_api=self._get_webview_api(),
            config=self.config,
            url=url,
            html=html,
            js_api=self.bridge,
        )

        return self._window

    def create_tray_controller(self) -> TrayController:
        """Create the tray controller connected to the application window."""
        if self._tray_controller is None:
            from seven.gui.tray import TrayController

            self._tray_controller = TrayController(
                on_show=self.show_window,
                on_exit=self.close,
                title=self.config.title,
            )

        return self._tray_controller

    def show_window(self) -> bool:
        """Restore and show the application window."""
        if self._closed or self._window is None:
            return False

        restore = getattr(self._window, "restore", None)

        if callable(restore):
            restore()

        show = getattr(self._window, "show", None)

        if not callable(show):
            raise RuntimeError("WebView window does not support show().")

        show()
        return True

    def _stop_owned_runtime(self) -> None:
        """Stop the runtime only when this application started it."""
        if not self._runtime_started_by_app:
            return

        self._runtime_started_by_app = False

        if self._runtime_manager is not None:
            self._runtime_manager.stop()

    def run(
        self,
        url: str | None = None,
        html: str | None = None,
        debug: bool = False,
    ) -> None:
        """Start the WebView application."""
        if self._closed:
            raise RuntimeError("Cannot run a closed application.")

        if self._started:
            raise RuntimeError("Application is already running.")

        webview_api = self._get_webview_api()

        try:
            if self._runtime_manager is not None:
                self._runtime_started_by_app = bool(
                    self._runtime_manager.start()
                )

            self.create_window(url=url, html=html)
            self._started = True
            webview_api.start(debug=debug)
        finally:
            self._closed = True
            self._stop_owned_runtime()

    def close(self) -> None:
        """Close the application window."""
        if self._closed:
            return

        if self._window is not None:
            self._window.destroy()

        self._closed = True
