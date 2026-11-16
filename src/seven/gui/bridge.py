
"""JavaScript-Python bridge for the S7VEN frontend."""

from __future__ import annotations
import asyncio
import base64
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from seven.core.voice.capture.audio_decoder import AudioDecoder
from seven.core.voice.capture.audio_validator import AudioValidator
from seven.core.voice.models import AudioFormat

if TYPE_CHECKING:
    from seven.core.voice.assistant import VoiceAssistant


class FrontendBridge:
    """Expose safe Python methods to the JavaScript frontend."""

    def __init__(
        self,
        voice_assistant_provider: Callable[[], VoiceAssistant] | None = None,
        voice_config_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._voice_assistant_provider = voice_assistant_provider
        self._voice_config_provider = voice_config_provider
        self._audio_decoder = AudioDecoder()
        self._audio_validator = AudioValidator()

    def _get_voice_config(self) -> Any | None:
        """Return the current voice configuration."""
        if self._voice_config_provider is None:
            return None

        return self._voice_config_provider()

    def get_voice_settings(self) -> dict[str, Any]:
        """Return public voice interaction settings."""
        config = self._get_voice_config()
        print(
            "VOICE CONFIG:",
            {
                "enabled": config.enabled,
                "input_enabled": config.input_enabled,
                "output_enabled": config.output_enabled,
            },
        )

        if config is None:
            return {
                "success": False,
                "error": "Voice configuration is unavailable.",
            }

        return {
            "success": True,
            "enabled": config.enabled,
            "input_enabled": config.input_enabled,
            "output_enabled": config.output_enabled,
            "speech_voice": config.speech_voice,
            "speech_rate": config.speech_rate,
            "wake_word_enabled": config.wake_word_enabled,
            "wake_word": config.wake_word,
            "vad_enabled": config.vad_enabled,
        }

    @staticmethod
    def _get_speech_speed(config: Any) -> float:
        """Convert configured speech rate into a TTS speed multiplier."""
        return config.speech_rate / 175.0

    def get_app_info(self) -> dict[str, str]:
        """Return basic application information."""
        return {
            "name": "S7VEN",
            "version": "0.1.0a0",
            "status": "ready",
        }

    def ping(self) -> dict[str, Any]:
        """Check whether the Python bridge is reachable."""
        return {
            "success": True,
            "message": "Python bridge is connected.",
        }

    def send_message(self, message: str) -> dict[str, Any]:
        """Receive and validate a message from the frontend."""
        if not isinstance(message, str):
            return {
                "success": False,
                "error": "Message must be a string.",
            }

        normalized_message = message.strip()

        if not normalized_message:
            return {
                "success": False,
                "error": "Message cannot be empty.",
            }

        return {
            "success": True,
            "message": normalized_message,
            "reply": (
                "Message received by S7VEN Core. "
                "AI response processing is not connected yet."
            ),
        }

    def get_voice_status(self) -> dict[str, Any]:
        """Return the current voice interaction status."""
        if self._voice_assistant_provider is None:
            return {
                "success": False,
                "enabled": False,
                "state": "unavailable",
                "error": "Voice assistant is not configured.",
            }

        try:
            assistant = self._voice_assistant_provider()

            return {
                "success": True,
                "enabled": True,
                "state": assistant.state.value,
                "interrupted": assistant.is_interrupted,
            }
        except Exception as exc:
            return {
                "success": False,
                "enabled": False,
                "state": "error",
                "error": str(exc),
            }

    def interrupt_voice(self) -> dict[str, Any]:
        """Interrupt the current voice operation."""
        if self._voice_assistant_provider is None:
            return {
                "success": False,
                "error": "Voice assistant is not configured.",
            }

        try:
            assistant = self._voice_assistant_provider()
            assistant.interrupt()

            return {
                "success": True,
                "state": assistant.state.value,
                "message": "Voice operation interrupted.",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }


def process_audio(
    self,
    payload: str,
    audio_format: str,
    sample_rate: int = 24000,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Process audio through the synchronous JavaScript bridge."""
    try:
        return asyncio.run(
            self._process_audio_async(
                payload=payload,
                audio_format=audio_format,
                sample_rate=sample_rate,
                conversation_id=conversation_id,
            )
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def _process_audio_async(
    self,
    payload: str,
    audio_format: str,
    sample_rate: int = 24000,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Process audio input and return an AI response with synthesized speech."""
    if self._voice_assistant_provider is None:
        return {
            "success": False,
            "error": "Voice assistant is not configured.",
        }

    try:
        config = self._get_voice_config()

        if config is not None:
            if not config.enabled:
                return {
                    "success": False,
                    "error": "Voice interaction is disabled.",
                }

            if not config.input_enabled:
                return {
                    "success": False,
                    "error": "Voice input is disabled.",
                }

        selected_format = AudioFormat(audio_format)
        audio = self._audio_decoder.decode(
            payload=payload,
            audio_format=selected_format,
            sample_rate=sample_rate,
        )
        self._audio_validator.validate(audio)

        assistant = self._voice_assistant_provider()

        transcription, response = await assistant.process_with_transcription(
            audio,
            conversation_id=conversation_id,
        )

        speech_audio = None

        if config is None or config.output_enabled:
            speech_audio = await assistant.synthesize_response(
                response,
                voice=config.speech_voice if config is not None else "alloy",
                speed=(
                    self._get_speech_speed(config)
                    if config is not None
                    else 1.0
                ),
            )

        return {
            "success": True,
            "transcription": transcription.text,
            "response": response.content,
            "audio": (
                base64.b64encode(speech_audio).decode("ascii")
                if speech_audio is not None
                else None
            ),
            "audio_format": "mp3" if speech_audio is not None else None,
            "model": response.model,
            "provider": response.provider,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }

    except (ValueError, TypeError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
