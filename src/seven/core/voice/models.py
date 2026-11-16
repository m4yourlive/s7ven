
"""Voice interaction data models for S7VEN."""

from dataclasses import dataclass
from enum import Enum


class AudioFormat(str, Enum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"


class VoiceState(str, Enum):
    """Voice interaction states."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"
    STARTED = "started"
    AUDIO_CAPTURED = "audio_captured"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    RESPONSE_STARTED = "response_started"
    RESPONSE_COMPLETED = "response_completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Represent captured audio input."""

    data: bytes
    format: AudioFormat
    sample_rate: int = 24000

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Audio data cannot be empty.")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Represent speech-to-text output."""

    text: str
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Transcription text cannot be empty.")


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    """Represent text-to-speech input."""

    text: str
    voice: str = "alloy"
    speed: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Speech text cannot be empty.")
        if not 0.25 <= self.speed <= 4.0:
            raise ValueError("Speech speed must be between 0.25 and 4.0.")
