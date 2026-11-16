"""Tests for voice activity monitoring."""

from unittest.mock import Mock

import pytest

from seven.core.voice.detection.interfaces import VoiceActivityDetector
from seven.core.voice.detection.voice_activity_monitor import (
    VoiceActivityMonitor,
)
from seven.core.voice.models import PCMFrame


@pytest.fixture
def detector():
    return Mock(spec=VoiceActivityDetector)


@pytest.fixture
def monitor(detector):
    return VoiceActivityMonitor(
        detector=detector,
        silence_duration_ms=600,
    )


def create_frame(duration_ms: int = 20) -> PCMFrame:
    sample_count = 48000 * duration_ms // 1000
    return PCMFrame(
        data=b"\x00\x00" * sample_count,
        sample_rate=48000,
        duration_ms=duration_ms,
    )


def test_detects_speech(monitor, detector):
    detector.is_speech.return_value = True

    assert monitor.process(create_frame()) is True


def test_tracks_silence_after_speech(monitor, detector):
    detector.is_speech.side_effect = [True, False, False]

    assert monitor.process(create_frame()) is True
    assert monitor.process(create_frame()) is True
    assert monitor.process(create_frame()) is True


def test_stops_after_silence_threshold(monitor, detector):
    detector.is_speech.side_effect = [True] + [False] * 30

    monitor.process(create_frame())

    result = None
    for _ in range(30):
        result = monitor.process(create_frame())

    assert result is False


def test_silence_without_speech_returns_false(monitor, detector):
    detector.is_speech.return_value = False

    assert monitor.process(create_frame()) is False


def test_reset_clears_monitor_state(monitor, detector):
    detector.is_speech.return_value = True
    monitor.process(create_frame())

    monitor.reset()

    detector.is_speech.return_value = False

    assert monitor.process(create_frame()) is False
    detector.reset.assert_called_once()


def test_rejects_invalid_silence_duration(detector):
    with pytest.raises(
        ValueError,
        match="Silence duration must be positive",
    ):
        VoiceActivityMonitor(detector, silence_duration_ms=0)