"""Audio input validation utilities."""

from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import AudioFormat, AudioInput


class AudioValidator:
    """Validate captured audio before processing."""

    DEFAULT_MAX_SIZE = 25 * 1024 * 1024
    SUPPORTED_FORMATS = {
        AudioFormat.WAV,
        AudioFormat.MP3,
        AudioFormat.WEBM,
    }

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        if max_size <= 0:
            raise ValueError("Maximum audio size must be positive.")

        self._max_size = max_size

    def validate(self, audio: AudioInput) -> None:
        """Validate audio format, size, and content."""
        if not isinstance(audio, AudioInput):
            raise AudioInputError("Invalid audio input.")

        if audio.format not in self.SUPPORTED_FORMATS:
            raise AudioInputError("Unsupported audio format.")

        if len(audio.data) > self._max_size:
            raise AudioInputError("Audio file exceeds maximum allowed size.")

        if not audio.data:
            raise AudioInputError("Audio data cannot be empty.")
