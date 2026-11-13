"""Tests for the GUI package structure."""


def test_gui_package_can_be_imported() -> None:
    """Ensure the GUI package is importable."""
    import seven.gui

    assert seven.gui is not None
