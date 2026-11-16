
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
class PCMFrame:
    """Represent a single 16-bit PCM audio frame."""

    data: bytes
    sample_rate: int
    duration_ms: int = 20

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("PCM frame data cannot be empty.")

        if self.sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("Unsupported PCM sample rate.")

        if self.duration_ms not in {10, 20, 30}:
            raise ValueError("PCM frame duration must be 10, 20, or 30 ms.")

        expected_size = (
            self.sample_rate * self.duration_ms // 1000 * 2
        )

        if len(self.data) != expected_size:
            raise ValueError("Invalid PCM frame data length.")

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

@dataclass(frozen=True, slots=True)
class WakeWordResult:
    """Represent a detected wake word."""

    detected: bool
    wake_word: str | None = None

    def __post_init__(self) -> None:
        if self.detected and not self.wake_word:
            raise ValueError(
                "Wake word must be provided when detection succeeds."
            )