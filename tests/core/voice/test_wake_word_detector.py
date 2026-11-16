"""Tests for basic wake word detection."""

from unittest.mock import MagicMock

from seven.core.voice.detection.wake_word_detector import (
    BasicWakeWordDetector,
)


def test_returns_false_when_no_wake_word_is_detected():
    detector = BasicWakeWordDetector()
    frame = MagicMock()

    assert detector.process(frame) is False


def test_reset_reactivates_detector():
    detector = BasicWakeWordDetector()

    detector.reset()

    assert detector.process(MagicMock()) is False

def test_detector_implements_wake_word_interface():
    from seven.core.voice.detection.interfaces import WakeWordDetector

    detector = BasicWakeWordDetector()

    assert isinstance(detector, WakeWordDetector)