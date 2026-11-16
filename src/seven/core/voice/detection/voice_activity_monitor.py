"""Monitor voice activity across consecutive audio frames."""

from seven.core.voice.detection.interfaces import VoiceActivityDetector
from seven.core.voice.models import PCMFrame


class VoiceActivityMonitor:
    """Track speech activity across consecutive audio frames."""

    def __init__(
        self,
        detector: VoiceActivityDetector,
        silence_duration_ms: int = 600,
    ) -> None:
        if silence_duration_ms <= 0:
            raise ValueError("Silence duration must be positive.")

        self._detector = detector
        self._silence_duration_ms = silence_duration_ms
        self._silence_elapsed_ms = 0
        self._speech_detected = False

    def process(self, frame: PCMFrame) -> bool:
        """Process a frame and report whether speech is currently active."""
        is_speech = self._detector.is_speech(frame)

        if is_speech:
            self._speech_detected = True
            self._silence_elapsed_ms = 0
            return True

        if self._speech_detected:
            self._silence_elapsed_ms += frame.duration_ms

        if self._silence_elapsed_ms >= self._silence_duration_ms:
            self._speech_detected = False
            self._silence_elapsed_ms = 0
            return False

        return self._speech_detected

    def reset(self) -> None:
        """Reset voice activity tracking."""
        self._silence_elapsed_ms = 0
        self._speech_detected = False
        self._detector.reset()