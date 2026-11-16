
"""OpenAI implementation of voice providers."""

from openai import APIError, AsyncOpenAI

from seven.core.voice.exceptions import (
    SpeechSynthesisError,
    TranscriptionError,
    VoiceProviderUnavailableError,
)
from seven.core.voice.interfaces.provider import (
    SpeechToTextProvider,
    TextToSpeechProvider,
)
from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    SpeechRequest,
    TranscriptionResult,
)


class OpenAIVoiceProvider(SpeechToTextProvider, TextToSpeechProvider):
    """Provide speech transcription and synthesis through OpenAI."""

    def __init__(
        self,
        api_key: str,
        transcription_model: str = "gpt-transcribe",
        speech_model: str = "gpt-4o-mini-tts",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise VoiceProviderUnavailableError(
                "OpenAI API key is required."
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
        )
        self._transcription_model = transcription_model
        self._speech_model = speech_model
        self._closed = False

    async def transcribe(
        self,
        audio: AudioInput,
    ) -> TranscriptionResult:
        """Convert audio input into text."""
        if self._closed:
            raise VoiceProviderUnavailableError(
                "Voice provider is closed."
            )

        extensions = {
            AudioFormat.WAV: ("audio.wav", "audio/wav"),
            AudioFormat.MP3: ("audio.mp3", "audio/mpeg"),
            AudioFormat.WEBM: ("audio.webm", "audio/webm"),
        }

        filename, content_type = extensions[audio.format]

        try:
            response = await self._client.audio.transcriptions.create(
                model=self._transcription_model,
                file=(filename, audio.data, content_type),
            )
            return TranscriptionResult(text=response.text)

        except APIError as exc:
            raise TranscriptionError(
                "OpenAI speech transcription failed."
            ) from exc

    async def synthesize(
        self,
        request: SpeechRequest,
    ) -> bytes:
        """Convert text into audio data."""
        if self._closed:
            raise VoiceProviderUnavailableError(
                "Voice provider is closed."
            )

        try:
            async with (
                self._client.audio.speech.with_streaming_response.create(
                    model=self._speech_model,
                    voice=request.voice,
                    input=request.text,
                    speed=request.speed,
                    response_format="mp3",
                ) as response
            ):
                chunks = [
                    chunk
                    async for chunk in response.iter_bytes()
                ]

            audio_data = b"".join(chunks)

            if not audio_data:
                raise SpeechSynthesisError(
                    "OpenAI returned empty audio data."
                )

            return audio_data

        except APIError as exc:
            raise SpeechSynthesisError(
                "OpenAI speech synthesis failed."
            ) from exc

    async def close(self) -> None:
        """Release OpenAI client resources."""
        if self._closed:
            return

        self._closed = True
        await self._client.close()
