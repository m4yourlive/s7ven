"""Tests for the WebView window factory."""

from unittest.mock import Mock

import pytest

from seven.gui.window import WindowConfig, create_webview_window


def test_default_window_config() -> None:
    config = WindowConfig()

    assert config.title == "S7VEN"
    assert config.width == 1280
    assert config.height == 800
    assert config.min_width == 960
    assert config.min_height == 600
    assert config.resizable is True
    assert config.fullscreen is False


def test_custom_window_config() -> None:
    config = WindowConfig(
        title="S7VEN Test",
        width=1440,
        height=900,
        min_width=1024,
        min_height=768,
        resizable=False,
        fullscreen=True,
    )

    assert config.title == "S7VEN Test"
    assert config.width == 1440
    assert config.height == 900
    assert config.min_width == 1024
    assert config.min_height == 768
    assert config.resizable is False
    assert config.fullscreen is True


def test_create_window_with_url() -> None:
    webview = Mock()
    config = WindowConfig()
    bridge = Mock()

    result = create_webview_window(
        webview,
        config,
        url="http://localhost:8000",
        js_api=bridge,
    )

    assert result is webview.create_window.return_value
    webview.create_window.assert_called_once_with(
        title=config.title,
        url="http://localhost:8000",
        html=None,
        width=config.width,
        height=config.height,
        min_size=(config.min_width, config.min_height),
        resizable=config.resizable,
        fullscreen=config.fullscreen,
        js_api=bridge,
    )


def test_create_window_with_html() -> None:
    webview = Mock()
    config = WindowConfig()
    html = "<h1>S7VEN</h1>"

    create_webview_window(
        webview,
        config,
        html=html,
    )

    kwargs = webview.create_window.call_args.kwargs

    assert kwargs["url"] is None
    assert kwargs["html"] == html


def test_create_window_with_custom_config() -> None:
    webview = Mock()
    config = WindowConfig(
        title="S7VEN Custom",
        width=1600,
        height=1000,
        min_width=1200,
        min_height=800,
        resizable=False,
        fullscreen=True,
    )

    create_webview_window(
        webview,
        config,
        url="http://localhost:8000",
    )

    kwargs = webview.create_window.call_args.kwargs

    assert kwargs["title"] == "S7VEN Custom"
    assert kwargs["width"] == 1600
    assert kwargs["height"] == 1000
    assert kwargs["min_size"] == (1200, 800)
    assert kwargs["resizable"] is False
    assert kwargs["fullscreen"] is True


def test_create_window_passes_javascript_api() -> None:
    webview = Mock()
    bridge = Mock()

    create_webview_window(
        webview,
        WindowConfig(),
        url="http://localhost:8000",
        js_api=bridge,
    )

    assert webview.create_window.call_args.kwargs["js_api"] is bridge


def test_create_window_rejects_url_and_html_together() -> None:
    with pytest.raises(ValueError):
        create_webview_window(
            Mock(),
            WindowConfig(),
            url="http://localhost:8000",
            html="<h1>S7VEN</h1>",
        )


@pytest.mark.parametrize(
    ("url", "html"),
    [
        ("", None),
        (None, ""),
    ],
)
def test_create_window_rejects_empty_source(
    url: str | None,
    html: str | None,
) -> None:
    with pytest.raises(ValueError):
        create_webview_window(
            Mock(),
            WindowConfig(),
            url=url,
            html=html,
        )
