
"""Tests for audio payload decoding."""

import base64

import pytest

from seven.core.voice.capture.audio_decoder import AudioDecoder
from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import AudioFormat


@pytest.fixture
def decoder():
    return AudioDecoder()


def test_decodes_valid_base64_audio(decoder):
    original = b"audio-data"
    payload = base64.b64encode(original).decode("ascii")

    result = decoder.decode(payload, AudioFormat.WAV)

    assert result.data == original
    assert result.format == AudioFormat.WAV
    assert result.sample_rate == 24000


def test_rejects_empty_payload(decoder):
    with pytest.raises(AudioInputError, match="cannot be empty"):
        decoder.decode("", AudioFormat.WAV)


def test_rejects_invalid_base64(decoder):
    with pytest.raises(AudioInputError, match="Invalid Base64"):
        decoder.decode("invalid-base64!", AudioFormat.WAV)


def test_rejects_unsupported_format(decoder):
    with pytest.raises(AudioInputError, match="Unsupported audio format"):
        decoder.decode(
            base64.b64encode(b"audio").decode("ascii"),
            "ogg",
        )


def test_rejects_empty_decoded_audio(decoder):
    payload = base64.b64encode(b"").decode("ascii")

    with pytest.raises(AudioInputError, match="Decoded audio data"):
        decoder.decode(payload, AudioFormat.WAV)


def test_rejects_payload_exceeding_maximum_size():
    decoder = AudioDecoder(max_size=4)
    payload = base64.b64encode(b"12345").decode("ascii")

    with pytest.raises(AudioInputError, match="exceeds maximum"):
        decoder.decode(payload, AudioFormat.WAV)


def test_rejects_invalid_sample_rate(decoder):
    payload = base64.b64encode(b"audio").decode("ascii")

    with pytest.raises(AudioInputError, match="metadata"):
        decoder.decode(payload, AudioFormat.WAV, sample_rate=0)
