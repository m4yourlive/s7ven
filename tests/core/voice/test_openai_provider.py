
"""Tests for the OpenAI voice provider."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    SpeechRequest,
)
from seven.core.voice.providers.openai_provider import OpenAIVoiceProvider


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.close = AsyncMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text="Hello S7VEN")
    )
    return client


@pytest.fixture
def provider(monkeypatch, mock_client):
    monkeypatch.setattr(
        "seven.core.voice.providers.openai_provider.AsyncOpenAI",
        MagicMock(return_value=mock_client),
    )

    return OpenAIVoiceProvider(api_key="test-api-key")


@pytest.mark.asyncio
async def test_transcribe_audio(provider, mock_client):
    audio = AudioInput(
        data=b"audio-data",
        format=AudioFormat.WAV,
    )

    result = await provider.transcribe(audio)

    assert result.text == "Hello S7VEN"
    mock_client.audio.transcriptions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesize_speech(provider, mock_client):
    async def iter_bytes():
        yield b"audio-"
        yield b"data"

    response = SimpleNamespace(iter_bytes=iter_bytes)

    @asynccontextmanager
    async def streaming_response(**kwargs):
        yield response

    mock_client.audio.speech.with_streaming_response.create = MagicMock(
        side_effect=streaming_response
    )

    result = await provider.synthesize(
        SpeechRequest(text="Hello S7VEN")
    )

    assert result == b"audio-data"


@pytest.mark.asyncio
async def test_close_provider(provider, mock_client):
    await provider.close()

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_is_idempotent(provider, mock_client):
    await provider.close()
    await provider.close()

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_after_close_raises_error(provider):
    from seven.core.voice.exceptions import VoiceProviderUnavailableError

    await provider.close()

    with pytest.raises(VoiceProviderUnavailableError):
        await provider.transcribe(
            AudioInput(
                data=b"audio-data",
                format=AudioFormat.WAV,
            )
        )


def test_provider_requires_api_key():
    from seven.core.voice.exceptions import VoiceProviderUnavailableError

    with pytest.raises(VoiceProviderUnavailableError):
        OpenAIVoiceProvider(api_key="")
