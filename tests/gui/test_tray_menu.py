"""Tests for the system tray menu."""

from __future__ import annotations

from typing import Any

import pytest

from seven.gui.tray_menu import create_tray_menu


class FakeMenuItem:
    """Fake pystray menu item."""

    def __init__(
        self,
        text: str,
        action: Any,
        *,
        default: bool = False,
    ) -> None:
        self.text = text
        self.action = action
        self.default = default


class FakeMenu:
    """Fake pystray menu."""

    def __init__(self, *items: FakeMenuItem) -> None:
        self.items = items


class FakePystray:
    """Fake pystray API."""

    MenuItem = FakeMenuItem
    Menu = FakeMenu


def create_callbacks() -> tuple[Any, Any, dict[str, int]]:
    """Create callback functions and invocation counters."""
    calls = {"show": 0, "exit": 0}

    def on_show() -> None:
        calls["show"] += 1

    def on_exit() -> None:
        calls["exit"] += 1

    return on_show, on_exit, calls


def test_create_tray_menu_returns_menu() -> None:
    on_show, on_exit, _ = create_callbacks()

    menu = create_tray_menu(
        FakePystray,
        on_show=on_show,
        on_exit=on_exit,
    )

    assert isinstance(menu, FakeMenu)


def test_menu_contains_expected_items() -> None:
    on_show, on_exit, _ = create_callbacks()

    menu = create_tray_menu(
        FakePystray,
        on_show=on_show,
        on_exit=on_exit,
    )

    assert len(menu.items) == 2
    assert menu.items[0].text == "Open S7VEN"
    assert menu.items[1].text == "Exit"


def test_open_item_is_default_action() -> None:
    on_show, on_exit, _ = create_callbacks()

    menu = create_tray_menu(
        FakePystray,
        on_show=on_show,
        on_exit=on_exit,
    )

    assert menu.items[0].default is True
    assert menu.items[1].default is False


def test_open_action_invokes_on_show() -> None:
    on_show, on_exit, calls = create_callbacks()

    menu = create_tray_menu(
        FakePystray,
        on_show=on_show,
        on_exit=on_exit,
    )

    menu.items[0].action(None, None)

    assert calls["show"] == 1
    assert calls["exit"] == 0


def test_exit_action_invokes_on_exit() -> None:
    on_show, on_exit, calls = create_callbacks()

    menu = create_tray_menu(
        FakePystray,
        on_show=on_show,
        on_exit=on_exit,
    )

    menu.items[1].action(None, None)

    assert calls["show"] == 0
    assert calls["exit"] == 1


def test_menu_actions_can_be_invoked_multiple_times() -> None:
    on_show, on_exit, calls = create_callbacks()

    menu = create_tray_menu(
        FakePystray,
        on_show=on_show,
        on_exit=on_exit,
    )

    menu.items[0].action(None, None)
    menu.items[0].action(None, None)
    menu.items[1].action(None, None)

    assert calls == {"show": 2, "exit": 1}


def test_create_menu_rejects_invalid_show_callback() -> None:
    _, on_exit, _ = create_callbacks()

    with pytest.raises(TypeError, match="on_show must be callable"):
        create_tray_menu(
            FakePystray,
            on_show=None,
            on_exit=on_exit,
        )


def test_create_menu_rejects_invalid_exit_callback() -> None:
    on_show, _, _ = create_callbacks()

    with pytest.raises(TypeError, match="on_exit must be callable"):
        create_tray_menu(
            FakePystray,
            on_show=on_show,
            on_exit=None,
        )
