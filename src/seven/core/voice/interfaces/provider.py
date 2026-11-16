
"""Abstract interfaces for voice providers."""

from abc import ABC, abstractmethod

from seven.core.voice.models import (
    AudioInput,
    SpeechRequest,
    TranscriptionResult,
)


class SpeechToTextProvider(ABC):
    """Interface for speech-to-text providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioInput,
    ) -> TranscriptionResult:
        """Convert audio input into text."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""


class TextToSpeechProvider(ABC):
    """Interface for text-to-speech providers."""

    @abstractmethod
    async def synthesize(
        self,
        request: SpeechRequest,
    ) -> bytes:
        """Convert text into audio data."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
