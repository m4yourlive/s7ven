
"""Tests for the voice interaction pipeline."""

from unittest.mock import AsyncMock

import pytest

from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    SpeechRequest,
    TranscriptionResult,
)
from seven.core.voice.pipeline import VoicePipeline


@pytest.fixture
def providers():
    stt = AsyncMock()
    tts = AsyncMock()

    stt.transcribe.return_value = TranscriptionResult(
        text="Hello S7VEN",
        language="en",
    )
    tts.synthesize.return_value = b"generated-audio"

    return stt, tts


@pytest.fixture
def pipeline(providers):
    stt, tts = providers
    return VoicePipeline(stt, tts)


@pytest.mark.asyncio
async def test_transcribe_audio(pipeline, providers):
    stt, _ = providers
    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )

    result = await pipeline.transcribe(audio)

    assert result.text == "Hello S7VEN"
    stt.transcribe.assert_awaited_once_with(audio)


@pytest.mark.asyncio
async def test_synthesize_speech(pipeline, providers):
    _, tts = providers
    request = SpeechRequest(text="Hello S7VEN")

    result = await pipeline.synthesize(request)

    assert result == b"generated-audio"
    tts.synthesize.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_close_providers(pipeline, providers):
    stt, tts = providers

    await pipeline.close()

    stt.close.assert_awaited_once()
    tts.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_is_idempotent(pipeline, providers):
    stt, tts = providers

    await pipeline.close()
    await pipeline.close()

    stt.close.assert_awaited_once()
    tts.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_after_close_raises_error(pipeline):
    await pipeline.close()

    with pytest.raises(RuntimeError, match="Voice pipeline is closed"):
        await pipeline.transcribe(
            AudioInput(
                data=b"audio-data",
                format=AudioFormat.WAV,
            )
        )


@pytest.mark.asyncio
async def test_synthesize_after_close_raises_error(pipeline):
    await pipeline.close()

    with pytest.raises(RuntimeError, match="Voice pipeline is closed"):
        await pipeline.synthesize(SpeechRequest(text="Hello"))
