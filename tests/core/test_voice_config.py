"""Tests for voice configuration."""

import pytest
from pydantic import ValidationError

from seven.core.config.config_schema import VoiceConfig


def test_voice_config_defaults():
    config = VoiceConfig()

    assert config.enabled is True
    assert config.input_enabled is True
    assert config.output_enabled is True
    assert config.speech_rate == 175
    assert config.speech_voice == "alloy"
    assert config.wake_word_enabled is False
    assert config.wake_word == "hey seven"
    assert config.vad_enabled is True
    assert config.vad_threshold == 500.0
    assert config.silence_duration_ms == 600
    assert config.max_audio_size_mb == 25


def test_voice_config_accepts_custom_values():
    config = VoiceConfig(
        enabled=True,
        speech_rate=200,
        speech_voice="nova",
        wake_word_enabled=True,
        wake_word="hello seven",
        vad_threshold=700.0,
        silence_duration_ms=1000,
        max_audio_size_mb=50,
    )

    assert config.enabled is True
    assert config.speech_rate == 200
    assert config.speech_voice == "nova"
    assert config.wake_word_enabled is True
    assert config.wake_word == "hello seven"
    assert config.vad_threshold == 700.0
    assert config.silence_duration_ms == 1000
    assert config.max_audio_size_mb == 50


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speech_rate", 50),
        ("speech_rate", 301),
        ("vad_threshold", 0),
        ("silence_duration_ms", 50),
        ("silence_duration_ms", 5001),
        ("max_audio_size_mb", 0),
        ("max_audio_size_mb", 101),
    ],
)
def test_voice_config_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        VoiceConfig(**{field: value})