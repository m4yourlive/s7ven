"""Voice assistant orchestration."""

from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.models import GenerationResponse
from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import AudioInput
from seven.core.voice.pipeline import VoicePipeline


class VoiceAssistant:
    """Coordinate voice input with AI response generation."""

    def __init__(
        self,
        voice_pipeline: VoicePipeline,
        ai_pipeline: AIResponsePipeline,
    ) -> None:
        self._voice_pipeline = voice_pipeline
        self._ai_pipeline = ai_pipeline

    async def process(
        self,
        audio: AudioInput,
        *,
        conversation_id: str | None = None,
    ) -> GenerationResponse:
        """Transcribe audio and generate an AI response."""
        if not isinstance(audio, AudioInput):
            raise AudioInputError("Invalid audio input.")

        transcription = await self._voice_pipeline.transcribe(audio)

        return await self._ai_pipeline.generate(
            transcription.text,
            conversation_id=conversation_id,
        )