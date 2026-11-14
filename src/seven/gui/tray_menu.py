"""System tray menu factory for S7VEN."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def create_tray_menu(
    pystray_api: Any,
    *,
    on_show: Callable[[], None],
    on_exit: Callable[[], None],
) -> Any:
    """Create the S7VEN system tray menu."""
    if not callable(on_show):
        raise TypeError("on_show must be callable.")

    if not callable(on_exit):
        raise TypeError("on_exit must be callable.")

    return pystray_api.Menu(
        pystray_api.MenuItem(
            "Open S7VEN",
            lambda icon, item: on_show(),
            default=True,
        ),
        pystray_api.MenuItem(
            "Exit",
            lambda icon, item: on_exit(),
        ),
    )
