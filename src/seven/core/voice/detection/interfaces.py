"""Interfaces for voice activity detection."""

from abc import ABC, abstractmethod

from seven.core.voice.models import PCMFrame


class VoiceActivityDetector(ABC):
    """Define the interface for voice activity detectors."""

    @abstractmethod
    def is_speech(self, frame: PCMFrame) -> bool:
        """Determine whether an audio frame contains speech."""

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""

class WakeWordDetector(ABC):
    """Define the interface for wake word detectors."""

    @abstractmethod
    def process(self, frame: PCMFrame) -> bool:
        """Process an audio frame and detect the configured wake word."""

    @abstractmethod
    def reset(self) -> None:
        """Reset wake word detection state."""