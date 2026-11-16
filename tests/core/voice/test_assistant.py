"""Tests for voice assistant orchestration."""

from unittest.mock import AsyncMock

import pytest

from seven.core.voice.assistant import VoiceAssistant
from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    TranscriptionResult,
)
from seven.core.voice.pipeline import VoicePipeline
from seven.core.ai.pipeline import AIResponsePipeline


@pytest.fixture
def assistant():
    voice_pipeline = AsyncMock(spec=VoicePipeline)
    ai_pipeline = AsyncMock(spec=AIResponsePipeline)

    return VoiceAssistant(
        voice_pipeline=voice_pipeline,
        ai_pipeline=ai_pipeline,
    ), voice_pipeline, ai_pipeline


@pytest.mark.asyncio
async def test_process_transcribes_and_generates_response(assistant):
    instance, voice_pipeline, ai_pipeline = assistant
    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )
    expected_response = object()

    voice_pipeline.transcribe.return_value = TranscriptionResult(
        text="Hello S7VEN"
    )
    ai_pipeline.generate.return_value = expected_response

    result = await instance.process(
        audio,
        conversation_id="conversation-123",
    )

    assert result is expected_response

    voice_pipeline.transcribe.assert_awaited_once_with(audio)
    ai_pipeline.generate.assert_awaited_once_with(
        "Hello S7VEN",
        conversation_id="conversation-123",
    )


@pytest.mark.asyncio
async def test_process_rejects_invalid_audio(assistant):
    instance, voice_pipeline, ai_pipeline = assistant

    with pytest.raises(AudioInputError):
        await instance.process(None)

    voice_pipeline.transcribe.assert_not_awaited()
    ai_pipeline.generate.assert_not_awaited()