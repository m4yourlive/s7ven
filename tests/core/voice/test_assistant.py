"""Tests for voice assistant orchestration."""

from unittest.mock import AsyncMock

import pytest
import asyncio

from seven.core.voice.assistant import VoiceAssistant
from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    TranscriptionResult,
)
from seven.core.voice.pipeline import VoicePipeline
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.voice.models import VoiceState


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

@pytest.mark.asyncio
async def test_process_resets_state_after_completion(assistant):
    instance, voice_pipeline, ai_pipeline = assistant

    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )
    voice_pipeline.transcribe.return_value = TranscriptionResult(
        text="Hello S7VEN"
    )

    await instance.process(audio)

    assert instance._turn_manager.state == VoiceState.IDLE


@pytest.mark.asyncio
async def test_process_resets_state_after_failure(assistant):
    instance, voice_pipeline, ai_pipeline = assistant
    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )
    voice_pipeline.transcribe.side_effect = RuntimeError("Transcription failed")

    with pytest.raises(RuntimeError, match="Transcription failed"):
        await instance.process(audio)

    assert instance._turn_manager.state == VoiceState.IDLE

@pytest.mark.asyncio
async def test_process_can_be_interrupted_during_transcription(assistant):
    instance, voice_pipeline, ai_pipeline = assistant
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def transcribe(audio):
        started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    voice_pipeline.transcribe.side_effect = transcribe

    task = asyncio.create_task(
        instance.process(
            AudioInput(
                data=b"audio",
                format=AudioFormat.WAV,
            )
        )
    )

    await started.wait()
    instance._turn_manager.interrupt()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    ai_pipeline.generate.assert_not_awaited()
    assert instance._turn_manager.state == VoiceState.IDLE

@pytest.mark.asyncio
async def test_process_can_be_interrupted_during_generation(assistant):
    instance, voice_pipeline, ai_pipeline = assistant
    started = asyncio.Event()
    cancelled = asyncio.Event()

    voice_pipeline.transcribe.return_value = TranscriptionResult(
        text="Hello S7VEN"
    )

    async def generate(*args, **kwargs):
        started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    ai_pipeline.generate.side_effect = generate

    task = asyncio.create_task(
        instance.process(
            AudioInput(
                data=b"audio",
                format=AudioFormat.WAV,
            )
        )
    )

    await started.wait()
    instance._turn_manager.interrupt()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    assert instance._turn_manager.state == VoiceState.IDLE


@pytest.mark.asyncio
async def test_synthesize_response(assistant):
    instance, voice_pipeline, ai_pipeline = assistant

    from seven.core.ai.models import GenerationResponse

    response = GenerationResponse(
        content="Hello S7VEN",
        model="test-model",
        provider="test-provider",
    )
    voice_pipeline.synthesize.return_value = b"speech-audio"

    result = await instance.synthesize_response(response)

    assert result == b"speech-audio"
    voice_pipeline.synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesize_response_rejects_invalid_response(assistant):
    instance, voice_pipeline, ai_pipeline = assistant

    with pytest.raises(TypeError):
        await instance.synthesize_response("invalid")

    voice_pipeline.synthesize.assert_not_awaited()
