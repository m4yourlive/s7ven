"""Tests for WebView application controller."""

from unittest.mock import Mock

import pytest

from seven.gui.application import WebViewApplication


def test_application_initialization() -> None:
    app = WebViewApplication(webview_api=Mock())

    assert app.window is None
    assert app.is_started is False
    assert app.is_closed is False


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


def test_close_application() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    window = app.create_window()

    app.close()

    window.destroy.assert_called_once()
    assert app.is_closed is True


def test_close_application_without_window() -> None:
    app = WebViewApplication(webview_api=Mock())

    app.close()

    assert app.is_closed is True


def test_close_application_only_once() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    window = app.create_window()

    app.close()
    app.close()

    window.destroy.assert_called_once()


def test_cannot_create_window_after_close() -> None:
    app = WebViewApplication(webview_api=Mock())

    app.close()

    with pytest.raises(RuntimeError):
        app.create_window()


def test_cannot_run_after_close() -> None:
    webview = Mock()
    app = WebViewApplication(webview_api=webview)

    app.close()

    with pytest.raises(RuntimeError):
        app.run()

    webview.start.assert_not_called()


def test_application_closes_after_start_failure() -> None:
    webview = Mock()
    webview.start.side_effect = RuntimeError("WebView startup failed")

    app = WebViewApplication(webview_api=webview)

    with pytest.raises(
        RuntimeError,
        match="WebView startup failed",
    ):
        app.run()

    assert app.is_started is True
    assert app.is_closed is True
