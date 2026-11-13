"""Tests for the JavaScript-Python bridge."""

from seven.gui.bridge import FrontendBridge


def test_get_app_info() -> None:
    bridge = FrontendBridge()

    result = bridge.get_app_info()

    assert result == {
        "name": "S7VEN",
        "version": "0.1.0a0",
        "status": "ready",
    }


def test_ping() -> None:
    bridge = FrontendBridge()

    result = bridge.ping()

    assert result["success"] is True
    assert result["message"] == "Python bridge is connected."


def test_send_message() -> None:
    bridge = FrontendBridge()

    result = bridge.send_message("Hello S7VEN")

    assert result["success"] is True
    assert result["message"] == "Hello S7VEN"
    assert "AI response processing" in result["reply"]


def test_send_message_strips_whitespace() -> None:
    bridge = FrontendBridge()

    result = bridge.send_message("  Hello S7VEN  ")

    assert result["success"] is True
    assert result["message"] == "Hello S7VEN"


def test_send_empty_message() -> None:
    bridge = FrontendBridge()

    result = bridge.send_message("   ")

    assert result["success"] is False
    assert result["error"] == "Message cannot be empty."


def test_send_non_string_message() -> None:
    bridge = FrontendBridge()

    result = bridge.send_message(None)

    assert result["success"] is False
    assert result["error"] == "Message must be a string."
