
"""Voice interaction package."""

from seven.core.voice.exceptions import (
    AudioInputError,
    SpeechSynthesisError,
    TranscriptionError,
    VoiceError,
    VoiceProviderUnavailableError,
)
from seven.core.voice.models import (
    AudioFormat,
    AudioInput,
    SpeechRequest,
    TranscriptionResult,
    VoiceState,
)

__all__ = [
    "AudioFormat",
    "AudioInput",
    "AudioInputError",
    "SpeechRequest",
    "SpeechSynthesisError",
    "TranscriptionError",
    "TranscriptionResult",
    "VoiceError",
    "VoiceProviderUnavailableError",
    "VoiceState",
]
