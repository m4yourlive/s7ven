"""S7VEN WebView application controller."""

from pathlib import Path
from typing import Any

from seven.gui.window import WindowConfig, create_webview_window


class WebViewApplication:
    """Manage the pywebview application lifecycle."""

    def __init__(
        self,
        config: WindowConfig | None = None,
        webview_api: Any | None = None,
    ) -> None:
        self.config = config or WindowConfig()
        self._webview_api = webview_api
        self._window: Any | None = None
        self._started = False
        self._closed = False

    @property
    def window(self) -> Any | None:
        """Return the current WebView window."""
        return self._window

    @property
    def is_started(self) -> bool:
        """Return whether the application has been started."""
        return self._started

    @property
    def is_closed(self) -> bool:
        """Return whether the application has been closed."""
        return self._closed

    @staticmethod
    def _get_frontend_url() -> str:
        """Resolve the local frontend entry point."""
        frontend_path = (
            Path(__file__).resolve().parent
            / "frontend"
            / "index.html"
        )

        if not frontend_path.is_file():
            raise FileNotFoundError(
                f"Frontend entry point not found: {frontend_path}"
            )

        return frontend_path.as_uri()

    def _get_webview_api(self) -> Any:
        """Load pywebview lazily when needed."""
        if self._webview_api is None:
            import webview

            self._webview_api = webview

        return self._webview_api

    def create_window(
        self,
        *,
        url: str | None = None,
        html: str | None = None,
    ) -> Any:
        """Create the main application window."""
        if self._closed:
            raise RuntimeError(
                "Cannot create a window after application closure."
            )

        if self._window is not None:
            raise RuntimeError(
                "Application window already exists."
            )

        if url is None and html is None:
            url = self._get_frontend_url()

        webview_api = self._get_webview_api()

        self._window = create_webview_window(
            webview_api,
            self.config,
            url=url,
            html=html,
        )

        return self._window

    def run(
        self,
        *,
        url: str | None = None,
        html: str | None = None,
        debug: bool = False,
    ) -> None:
        """Create the main window and start the WebView event loop."""
        if self._closed:
            raise RuntimeError(
                "Cannot run a closed application."
            )

        if self._started:
            raise RuntimeError(
                "Application has already been started."
            )

        webview_api = self._get_webview_api()

        self.create_window(
            url=url,
            html=html,
        )

        self._started = True

        try:
            webview_api.start(debug=debug)
        finally:
            self._closed = True

    def close(self) -> None:
        """Close the application window."""
        if self._closed:
            return

        if self._window is not None:
            self._window.destroy()

        self._closed = True
