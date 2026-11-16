
"""Voice interaction pipeline."""

from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.interfaces.provider import (
    SpeechToTextProvider,
    TextToSpeechProvider,
)
from seven.core.voice.models import (
    AudioInput,
    SpeechRequest,
    TranscriptionResult,
)


class VoicePipeline:
    """Coordinate speech transcription and synthesis."""

    def __init__(
        self,
        stt_provider: SpeechToTextProvider,
        tts_provider: TextToSpeechProvider,
    ) -> None:
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._closed = False

    async def transcribe(
        self,
        audio: AudioInput,
    ) -> TranscriptionResult:
        """Transcribe captured audio into text."""
        self._ensure_open()

        if not isinstance(audio, AudioInput):
            raise AudioInputError("Invalid audio input.")

        return await self._stt_provider.transcribe(audio)

    async def synthesize(
        self,
        request: SpeechRequest,
    ) -> bytes:
        """Synthesize speech from text."""
        self._ensure_open()
        return await self._tts_provider.synthesize(request)

    async def close(self) -> None:
        """Release voice provider resources."""
        if self._closed:
            return

        self._closed = True
        await self._stt_provider.close()

        if self._tts_provider is not self._stt_provider:
            await self._tts_provider.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Voice pipeline is closed.")
