"""Manage voice conversation turns and state transitions."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from seven.core.voice.models import VoiceState

T = TypeVar("T")


class VoiceTurnManager:
    """Manage the state and interruption of a voice conversation."""

    def __init__(self) -> None:
        self._state = VoiceState.IDLE
        self._interruption_event = asyncio.Event()

    @property
    def state(self) -> VoiceState:
        """Return the current voice state."""
        return self._state

    @property
    def is_interrupted(self) -> bool:
        """Return whether the current turn was interrupted."""
        return self._interruption_event.is_set()

    def start_listening(self) -> None:
        """Start listening for user audio."""
        self._interruption_event.clear()
        self._state = VoiceState.LISTENING

    def start_processing(self) -> None:
        """Start processing captured audio."""
        self._state = VoiceState.PROCESSING

    def start_speaking(self) -> None:
        """Start assistant speech playback."""
        self._state = VoiceState.SPEAKING

    def interrupt(self) -> None:
        """Signal interruption of the current voice operation."""
        self._interruption_event.set()
        self._state = VoiceState.IDLE

    async def wait_for_interruption(self) -> None:
        """Wait until the current turn is interrupted."""
        await self._interruption_event.wait()

    async def run_interruptible(
        self,
        awaitable: Awaitable[T],
    ) -> T:
        """Run an asynchronous operation that can be interrupted."""
        operation = asyncio.ensure_future(awaitable)
        interruption = asyncio.create_task(
            self.wait_for_interruption()
        )

        try:
            done, _ = await asyncio.wait(
                {operation, interruption},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if interruption in done:
                operation.cancel()
                await asyncio.gather(
                    operation,
                    return_exceptions=True,
                )
                raise asyncio.CancelledError(
                    "Voice operation was interrupted."
                )

            return await operation

        except asyncio.CancelledError:
            operation.cancel()
            await asyncio.gather(
                operation,
                return_exceptions=True,
            )
            raise

        finally:
            interruption.cancel()
            await asyncio.gather(
                interruption,
                return_exceptions=True,
            )

    def reset(self) -> None:
        """Reset the conversation state."""
        self._interruption_event.clear()
        self._state = VoiceState.IDLE
