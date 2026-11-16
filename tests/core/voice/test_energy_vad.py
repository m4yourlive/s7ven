"""Tests for energy-based voice activity detection."""

import struct

import pytest

from seven.core.voice.detection import EnergyVoiceActivityDetector
from seven.core.voice.models import PCMFrame


@pytest.fixture
def detector():
    return EnergyVoiceActivityDetector(threshold=500.0)


def create_frame(amplitude: int) -> PCMFrame:
    samples = [amplitude] * 480
    data = struct.pack("<480h", *samples)

    return PCMFrame(
        data=data,
        sample_rate=48000,
        duration_ms=10,
    )


def test_detects_speech(detector):
    assert detector.is_speech(create_frame(1000)) is True


def test_detects_silence(detector):
    assert detector.is_speech(create_frame(100)) is False


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold must be positive"):
        EnergyVoiceActivityDetector(threshold=0)


def test_reset_does_not_raise(detector):
    detector.reset()