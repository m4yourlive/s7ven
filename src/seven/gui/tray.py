"""S7VEN system tray controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from seven.gui.tray_menu import create_tray_menu


class TrayController:
    """Manage the S7VEN system tray icon and its lifecycle."""

    def __init__(
        self,
        *,
        on_show: Callable[[], None],
        on_exit: Callable[[], None],
        title: str = "S7VEN",
        icon_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not callable(on_show):
            raise TypeError("on_show must be callable.")

        if not callable(on_exit):
            raise TypeError("on_exit must be callable.")

        if not title.strip():
            raise ValueError("Tray title cannot be empty.")

        self._on_show = on_show
        self._on_exit = on_exit
        self._title = title
        self._icon_factory = icon_factory or self._create_icon
        self._icon: Any | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the tray icon is running."""
        return self._icon is not None

    def _create_icon(self) -> Any:
        """Create the native system tray icon."""
        import pystray
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text(
            (21, 10),
            "7",
            fill="#ff2d2d",
            stroke_width=1,
        )

        menu = create_tray_menu(
            pystray,
            on_show=self._on_show,
            on_exit=self._on_exit,
        )

        return pystray.Icon(
            "s7ven",
            image,
            self._title,
            menu,
        )

    def start(self) -> None:
        """Start the system tray icon."""
        if self._icon is not None:
            raise RuntimeError("System tray is already running.")

        icon = self._icon_factory()
        icon.run_detached()
        self._icon = icon

    def stop(self) -> None:
        """Stop the system tray icon."""
        if self._icon is None:
            return

        icon = self._icon
        self._icon = None
        icon.stop()
