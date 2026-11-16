"""Tests for text-to-speech functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.voice.exceptions import VoiceProviderUnavailableError
from seven.core.voice.models import SpeechRequest
from seven.core.voice.providers.openai_provider import OpenAIVoiceProvider


@pytest.fixture
def provider():
    instance = OpenAIVoiceProvider(api_key="test-api-key")
    instance._client = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_synthesize_returns_audio_bytes(provider):
    response = MagicMock()

    async def audio_chunks():
        yield b"audio"
        yield b"data"

    response.iter_bytes = audio_chunks

    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=response)
    context_manager.__aexit__ = AsyncMock(return_value=None)

    provider._client.audio.speech.with_streaming_response.create = (
        MagicMock(return_value=context_manager)
    )

    result = await provider.synthesize(SpeechRequest(text="Hello"))

    assert result == b"audiodata"


@pytest.mark.asyncio
async def test_synthesize_rejects_closed_provider(provider):
    await provider.close()

    with pytest.raises(VoiceProviderUnavailableError):
        await provider.synthesize(SpeechRequest(text="Hello"))