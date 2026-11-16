
"""Decode audio payloads received from the frontend."""

import base64
import binascii

from seven.core.voice.exceptions import AudioInputError
from seven.core.voice.models import AudioFormat, AudioInput


class AudioDecoder:
    """Decode Base64 audio payloads into audio input objects."""

    def __init__(
        self,
        max_size: int = 25 * 1024 * 1024,
    ) -> None:
        if max_size <= 0:
            raise ValueError("Maximum audio size must be positive.")

        self._max_size = max_size

    def decode(
        self,
        payload: str,
        audio_format: AudioFormat,
        sample_rate: int = 24000,
    ) -> AudioInput:
        """Decode a Base64 audio payload."""
        if not isinstance(payload, str):
            raise AudioInputError("Audio payload must be a string.")

        if not isinstance(audio_format, AudioFormat):
            raise AudioInputError("Unsupported audio format.")

        try:
            encoded = payload.encode("ascii")
            data = base64.b64decode(encoded, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise AudioInputError("Invalid Base64 audio payload.") from exc

        if not data:
            raise AudioInputError("Decoded audio data cannot be empty.")

        if len(data) > self._max_size:
            raise AudioInputError("Audio file exceeds maximum allowed size.")

        try:
            return AudioInput(
                data=data,
                format=audio_format,
                sample_rate=sample_rate,
            )
        except ValueError as exc:
            raise AudioInputError("Invalid audio input metadata.") from exc
