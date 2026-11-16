"""Tests for the JavaScript-Python bridge."""

from seven.gui.bridge import FrontendBridge
from unittest.mock import Mock

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


def test_interrupt_voice_without_assistant() -> None:
    bridge = FrontendBridge()

    result = bridge.interrupt_voice()

    assert result["success"] is False
    assert result["error"] == "Voice assistant is not configured."


def test_interrupt_voice_success() -> None:
    assistant = Mock()
    assistant.state.value = "idle"

    bridge = FrontendBridge(
        voice_assistant_provider=lambda: assistant,
    )

    result = bridge.interrupt_voice()

    assistant.interrupt.assert_called_once()
    assert result["success"] is True
    assert result["state"] == "idle"


def test_interrupt_voice_provider_error() -> None:
    def raise_error():
        raise RuntimeError("Voice provider failed.")

    bridge = FrontendBridge(
        voice_assistant_provider=raise_error,
    )

    result = bridge.interrupt_voice()

    assert result["success"] is False
    assert result["error"] == "Voice provider failed."


def test_get_voice_status_with_assistant() -> None:
    from unittest.mock import Mock

    assistant = Mock()
    assistant.state.value = "processing"
    assistant.is_interrupted = False

    bridge = FrontendBridge(
        voice_assistant_provider=lambda: assistant,
    )

    result = bridge.get_voice_status()

    assert result == {
        "success": True,
        "enabled": True,
        "state": "processing",
        "interrupted": False,
    }


def test_get_voice_status_provider_error() -> None:
    def raise_error():
        raise RuntimeError("Voice provider failed.")

    bridge = FrontendBridge(
        voice_assistant_provider=raise_error,
    )

    result = bridge.get_voice_status()

    assert result["success"] is False
    assert result["state"] == "error"
    assert result["error"] == "Voice provider failed."


def test_get_voice_settings_returns_configured_values() -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(
        enabled=True,
        input_enabled=True,
        output_enabled=False,
        speech_voice="alloy",
        speech_rate=175,
        wake_word_enabled=False,
        wake_word="hey seven",
        vad_enabled=True,
    )

    bridge = FrontendBridge(
        voice_config_provider=lambda: config,
    )

    result = bridge.get_voice_settings()

    assert result["success"] is True
    assert result["enabled"] is True
    assert result["input_enabled"] is True
    assert result["output_enabled"] is False
    assert result["speech_voice"] == "alloy"
    assert result["speech_rate"] == 175
    assert result["wake_word_enabled"] is False
    assert result["vad_enabled"] is True


def test_get_speech_speed_converts_speech_rate() -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(speech_rate=175)

    assert FrontendBridge._get_speech_speed(config) == 1.0
