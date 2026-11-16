"""Basic wake word detection implementation."""

from seven.core.voice.detection.interfaces import WakeWordDetector
from seven.core.voice.models import PCMFrame


class BasicWakeWordDetector(WakeWordDetector):
    """Provide a placeholder wake word detector."""

    def __init__(self) -> None:
        self._active = True

    def process(self, frame: PCMFrame) -> bool:
        """Process an audio frame for wake word detection."""
        if not self._active:
            return False

        return False

    def reset(self) -> None:
        """Reset wake word detection state."""
        self._active = True