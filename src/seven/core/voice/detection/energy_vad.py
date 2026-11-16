"""Energy-based voice activity detection."""

import math
import struct

from seven.core.voice.detection.interfaces import VoiceActivityDetector
from seven.core.voice.models import PCMFrame


class EnergyVoiceActivityDetector(VoiceActivityDetector):
    """Detect voice activity using audio signal energy."""

    def __init__(self, threshold: float = 500.0) -> None:
        if threshold <= 0:
            raise ValueError("Threshold must be positive.")

        self._threshold = threshold

    def is_speech(self, frame: PCMFrame) -> bool:
        """Determine whether an audio frame contains speech."""
        samples = struct.unpack(
            f"<{len(frame.data) // 2}h",
            frame.data,
        )

        if not samples:
            return False

        rms = math.sqrt(
            sum(sample * sample for sample in samples) / len(samples)
        )

        return rms >= self._threshold

    def reset(self) -> None:
        """Reset detector state."""
        return None