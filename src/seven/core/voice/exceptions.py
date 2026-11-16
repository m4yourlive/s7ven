
"""Exceptions for voice interaction."""

from seven.core.ai.exceptions import AIError


class VoiceError(AIError):
    """Base exception for voice-related failures."""


class VoiceProviderUnavailableError(VoiceError):
    """Raised when a voice provider is unavailable."""


class AudioInputError(VoiceError):
    """Raised when audio input is invalid or cannot be processed."""


class TranscriptionError(VoiceError):
    """Raised when speech transcription fails."""


class SpeechSynthesisError(VoiceError):
    """Raised when speech synthesis fails."""
