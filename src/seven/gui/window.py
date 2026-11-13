"""WebView window configuration and factory."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Configuration for the S7VEN desktop window."""

    title: str = "S7VEN"
    width: int = 1280
    height: int = 800
    min_width: int = 960
    min_height: int = 600
    resizable: bool = True
    fullscreen: bool = False

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Window title cannot be empty.")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Window dimensions must be positive.")

        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("Minimum dimensions must be positive.")

        if self.width < self.min_width:
            raise ValueError("Width cannot be smaller than min_width.")

        if self.height < self.min_height:
            raise ValueError("Height cannot be smaller than min_height.")


def create_webview_window(
    webview_api: Any,
    config: WindowConfig,
    *,
    url: str | None = None,
    html: str | None = None,
) -> Any:
    """Create a native window through the pywebview API."""

    if url is not None and html is not None:
        raise ValueError("Specify either url or html, not both.")

    if url is not None and not url.strip():
        raise ValueError("URL cannot be empty.")

    if html is not None and not html.strip():
        raise ValueError("HTML content cannot be empty.")

    return webview_api.create_window(
        title=config.title,
        url=url,
        html=html,
        width=config.width,
        height=config.height,
        min_size=(config.min_width, config.min_height),
        resizable=config.resizable,
        fullscreen=config.fullscreen,
    )
