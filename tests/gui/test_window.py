"""Tests for WebView application controller."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from seven.gui.application import WebViewApplication


def test_application_initialization() -> None:
    app = WebViewApplication(webview_api=Mock())

    assert app.window is None
    assert app.is_started is False


def test_create_window() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    window = app.create_window(
        url="http://localhost:8000",
    )

    assert window is webview.create_window.return_value
    webview.create_window.assert_called_once()


def test_create_window_uses_default_frontend() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    app.create_window()

    kwargs = webview.create_window.call_args.kwargs

    assert kwargs["url"].startswith("file:")
    assert kwargs["url"].endswith("index.html")
    assert kwargs["html"] is None


def test_default_frontend_exists() -> None:
    frontend_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "seven"
        / "gui"
        / "frontend"
        / "index.html"
    )

    assert frontend_path.is_file()


def test_create_window_with_html() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    app.create_window(html="<h1>S7VEN</h1>")

    kwargs = webview.create_window.call_args.kwargs

    assert kwargs["html"] == "<h1>S7VEN</h1>"
    assert kwargs["url"] is None


def test_create_window_only_once() -> None:
    app = WebViewApplication(webview_api=Mock())

    app.create_window()

    with pytest.raises(RuntimeError):
        app.create_window()


def test_run_application() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    app.run(
        url="http://localhost:8000",
        debug=True,
    )

    assert app.is_started is True
    assert app.window is not None

    webview.start.assert_called_once_with(debug=True)


def test_run_application_with_default_frontend() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    app.run()

    assert app.is_started is True
    assert app.window is not None

    kwargs = webview.create_window.call_args.kwargs

    assert kwargs["url"].endswith("index.html")
    webview.start.assert_called_once_with(debug=False)


def test_run_application_only_once() -> None:
    app = WebViewApplication(webview_api=Mock())

    app.run()

    with pytest.raises(RuntimeError):
        app.run()
