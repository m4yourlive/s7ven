
"""Tests for audio input validation."""

import pytest

from seven.core.voice.capture import AudioValidator
from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import AudioFormat, AudioInput


@pytest.fixture
def validator():
    return AudioValidator()


def test_accepts_valid_audio(validator):
    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )

    assert validator.validate(audio) is None


def test_rejects_invalid_input(validator):
    with pytest.raises(AudioInputError, match="Invalid audio input"):
        validator.validate(b"invalid")


def test_rejects_unsupported_format():
    validator = AudioValidator()

    with pytest.raises(AudioInputError, match="Unsupported audio format"):
        validator.validate(
            AudioInput(
                data=b"audio-data",
                format="ogg",
            )
        )


def test_rejects_audio_exceeding_maximum_size():
    validator = AudioValidator(max_size=4)

    with pytest.raises(AudioInputError, match="exceeds maximum"):
        validator.validate(
            AudioInput(
                data=b"12345",
                format=AudioFormat.WAV,
            )
        )


def test_rejects_non_positive_max_size():
    with pytest.raises(ValueError, match="must be positive"):
        AudioValidator(max_size=0)
