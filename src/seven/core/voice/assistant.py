
"""Voice assistant orchestration."""

from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.models import GenerationResponse
from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import AudioInput, TranscriptionResult, VoiceState
from seven.core.voice.pipeline import VoicePipeline
from seven.core.voice.turn_manager import VoiceTurnManager


class VoiceAssistant:
    """Coordinate voice input with AI response generation."""

    def __init__(
        self,
        voice_pipeline: VoicePipeline,
        ai_pipeline: AIResponsePipeline,
        turn_manager: VoiceTurnManager | None = None,
    ) -> None:
        self._voice_pipeline = voice_pipeline
        self._ai_pipeline = ai_pipeline
        self._turn_manager = turn_manager or VoiceTurnManager()

    @property
    def state(self) -> VoiceState:
        """Return the current voice interaction state."""
        return self._turn_manager.state

    @property
    def is_interrupted(self) -> bool:
        """Return whether the current turn was interrupted."""
        return self._turn_manager.is_interrupted

    def interrupt(self) -> None:
        """Interrupt the current voice operation."""
        self._turn_manager.interrupt()

    async def process(
        self,
        audio: AudioInput,
        *,
        conversation_id: str | None = None,
    ) -> GenerationResponse:
        """Transcribe audio and generate an AI response."""
        _, response = await self.process_with_transcription(
            audio,
            conversation_id=conversation_id,
        )
        return response


    async def process_with_transcription(
        self,
        audio: AudioInput,
        *,
        conversation_id: str | None = None,
    ) -> tuple[TranscriptionResult, GenerationResponse]:
        """Transcribe audio and return both transcription and AI response."""
        if not isinstance(audio, AudioInput):
            raise AudioInputError("Invalid audio input.")

        self._turn_manager.reset()
        self._turn_manager.start_processing()

        try:
            transcription = await self._turn_manager.run_interruptible(
                self._voice_pipeline.transcribe(audio)
            )

            response = await self._turn_manager.run_interruptible(
                self._ai_pipeline.generate(
                    transcription.text,
                    conversation_id=conversation_id,
                )
            )

            return transcription, response
        finally:
            if not self._turn_manager.is_interrupted:
                self._turn_manager.reset()

    async def synthesize_response(
        self,
        response: GenerationResponse,
        *,
        voice: str = "alloy",
        speed: float = 1.0,
    ) -> bytes:
        """Convert an AI response into speech audio."""
        from seven.core.voice.models import SpeechRequest

        if not isinstance(response, GenerationResponse):
            raise TypeError("response must be a GenerationResponse.")

        self._turn_manager.start_speaking()

        try:
            return await self._turn_manager.run_interruptible(
                self._voice_pipeline.synthesize(
                    SpeechRequest(
                        text=response.content,
                        voice=voice,
                        speed=speed,
                    )
                )
            )
        finally:
            if not self._turn_manager.is_interrupted:
                self._turn_manager.reset()

