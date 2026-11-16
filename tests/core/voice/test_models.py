"""Tests for voice interaction models."""

import pytest

from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    SpeechRequest,
    TranscriptionResult,
    VoiceState,
)


def test_audio_input_accepts_valid_data():
    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )

    assert audio.data == b"audio-data"
    assert audio.format == AudioFormat.WAV
    assert audio.sample_rate == 24000


def test_audio_input_rejects_empty_data():
    with pytest.raises(ValueError, match="Audio data cannot be empty"):
        AudioInput(data=b"", format=AudioFormat.WAV)


def test_audio_input_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        AudioInput(
            data=b"audio-data",
            format=AudioFormat.WAV,
            sample_rate=0,
        )


def test_transcription_result_accepts_valid_text():
    result = TranscriptionResult(
        text="Hello S7VEN",
        language="en",
    )

    assert result.text == "Hello S7VEN"
    assert result.language == "en"


def test_transcription_result_rejects_empty_text():
    with pytest.raises(
        ValueError,
        match="Transcription text cannot be empty",
    ):
        TranscriptionResult(text="   ")


def test_speech_request_accepts_valid_values():
    request = SpeechRequest(
        text="Hello S7VEN",
        voice="alloy",
        speed=1.25,
    )

    assert request.text == "Hello S7VEN"
    assert request.voice == "alloy"
    assert request.speed == 1.25


@pytest.mark.parametrize("speed", [0.1, 4.1])
def test_speech_request_rejects_invalid_speed(speed):
    with pytest.raises(
        ValueError,
        match="Speech speed must be between",
    ):
        SpeechRequest(text="Hello", speed=speed)


def test_voice_state_values():
    assert VoiceState.IDLE.value == "idle"
    assert VoiceState.LISTENING.value == "listening"
    assert VoiceState.PROCESSING.value == "processing"
    assert VoiceState.SPEAKING.value == "speaking"
    assert VoiceState.ERROR.value == "error"
