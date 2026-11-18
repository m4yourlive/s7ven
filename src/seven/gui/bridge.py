"""JavaScript-Python bridge for the S7VEN frontend."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Callable
from typing import Any

from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.conversation_service import ConversationService
from seven.core.conversations.memory_consolidator import MemoryConsolidator
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.message_service import MessageService
from seven.core.database.session import DatabaseSessionManager
from seven.core.voice.assistant import VoiceAssistant
from seven.core.voice.capture.audio_decoder import AudioDecoder
from seven.core.voice.capture.audio_validator import AudioValidator
from seven.core.voice.models import AudioFormat


logger = logging.getLogger(__name__)


class FrontendBridge:
    """Expose safe Python methods to the JavaScript frontend."""

    def __init__(
        self,
        voice_assistant_provider: Callable[[], VoiceAssistant] | None = None,
        voice_config_provider: Callable[[], Any] | None = None,
        ai_pipeline_provider: Callable[[], Any] | None = None,
        database_session_manager_provider: (
            Callable[[], DatabaseSessionManager] | None
        ) = None,
    ) -> None:
        self._voice_assistant_provider = voice_assistant_provider
        self._voice_config_provider = voice_config_provider
        self._ai_pipeline_provider = ai_pipeline_provider
        self._database_session_manager_provider = (
            database_session_manager_provider
        )

        self._audio_decoder = AudioDecoder()
        self._audio_validator = AudioValidator()

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # AI conversation
    # ------------------------------------------------------------------

    def send_message(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Receive a message and process it through the AI pipeline."""
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

        if (
            self._ai_pipeline_provider is None
            or self._database_session_manager_provider is None
        ):
            return {
                "success": True,
                "message": normalized_message,
                "reply": (
                    "Message received by S7VEN Core. "
                    "AI response processing is not connected yet."
                ),
            }

        try:
            return asyncio.run(
                self._send_message_async(
                    message=normalized_message,
                    conversation_id=conversation_id,
                )
            )
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    async def _send_message_async(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process and persist an AI conversation message."""
        if self._ai_pipeline_provider is None:
            raise RuntimeError("AI pipeline is not configured.")

        if self._database_session_manager_provider is None:
            raise RuntimeError("Database session manager is not configured.")

        pipeline = self._ai_pipeline_provider()
        session_manager = self._database_session_manager_provider()

        async with session_manager.session() as session:
            try:
                conversation_service = ConversationService(session)

                if conversation_id:
                    conversation = await conversation_service.get(
                        conversation_id
                    )

                    if conversation is None:
                        raise ValueError("Conversation was not found.")
                else:
                    conversation = await conversation_service.create(
                        title=message[:80] or "New Conversation"
                    )
                    conversation_id = str(conversation.id)

                message_service = MessageService(session)
                memory_service = MemoryService(session)
                memory_retriever = MemoryRetriever(memory_service)
                context_injector = ContextInjector(memory_retriever)

                response = await pipeline.generate(
                    message,
                    message_service=message_service,
                    conversation_id=conversation_id,
                    context_injector=context_injector,
                )

                try:
                    async with session.begin_nested():
                        memory_consolidator = MemoryConsolidator(
                            message_service=message_service,
                            memory_service=memory_service,
                        )

                        await memory_consolidator.consolidate(
                            conversation_id
                        )

                except Exception:
                    logger.exception(
                        "Automatic memory consolidation failed."
                    )

                await session.commit()

                return {
                    "success": True,
                    "message": message,
                    "reply": response.content,
                    "conversation_id": conversation_id,
                    "model": response.model,
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                }

            except Exception:
                await session.rollback()
                raise

    # ------------------------------------------------------------------
    # Voice configuration
    # ------------------------------------------------------------------

    def _get_voice_config(self) -> Any | None:
        """Return the current voice configuration."""
        if self._voice_config_provider is None:
            return None

        return self._voice_config_provider()

    def get_voice_settings(self) -> dict[str, Any]:
        """Return public voice interaction settings."""
        config = self._get_voice_config()

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

    # ------------------------------------------------------------------
    # Voice assistant
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

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
            return {
                "success": False,
                "error": str(exc),
            }

    async def _process_audio_async(
        self,
        payload: str,
        audio_format: str,
        sample_rate: int = 24000,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process audio input and return an AI response with speech."""
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

            transcription, response = (
                await assistant.process_with_transcription(
                    audio,
                    conversation_id=conversation_id,
                )
            )

            speech_audio = None

            if config is None or config.output_enabled:
                speech_audio = await assistant.synthesize_response(
                    response,
                    voice=(
                        config.speech_voice
                        if config is not None
                        else "alloy"
                    ),
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
                "audio_format": (
                    "mp3" if speech_audio is not None else None
                ),
                "model": response.model,
                "provider": response.provider,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            }

        except (ValueError, TypeError) as exc:
            return {
                "success": False,
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }