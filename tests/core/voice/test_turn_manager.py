"""Tests for voice turn management."""

from seven.core.voice.models import VoiceState
from seven.core.voice.turn_manager import VoiceTurnManager
import asyncio
import pytest

def test_initial_state_is_idle():
    manager = VoiceTurnManager()

    assert manager.state == VoiceState.IDLE


def test_start_listening():
    manager = VoiceTurnManager()

    manager.start_listening()

    assert manager.state == VoiceState.LISTENING


def test_start_processing():
    manager = VoiceTurnManager()

    manager.start_processing()

    assert manager.state == VoiceState.PROCESSING


def test_start_speaking():
    manager = VoiceTurnManager()

    manager.start_speaking()

    assert manager.state == VoiceState.SPEAKING


def test_interrupt_returns_to_idle():
    manager = VoiceTurnManager()
    manager.start_speaking()

    manager.interrupt()

    assert manager.state == VoiceState.IDLE


def test_reset_returns_to_idle():
    manager = VoiceTurnManager()
    manager.start_processing()

    manager.reset()

    assert manager.state == VoiceState.IDLE

def test_interrupt_sets_interruption_signal():
    manager = VoiceTurnManager()

    manager.interrupt()

    assert manager.is_interrupted is True


def test_start_listening_clears_interruption_signal():
    manager = VoiceTurnManager()
    manager.interrupt()

    manager.start_listening()

    assert manager.is_interrupted is False


@pytest.mark.asyncio
async def test_wait_for_interruption():
    manager = VoiceTurnManager()

    task = asyncio.create_task(manager.wait_for_interruption())
    manager.interrupt()

    await asyncio.wait_for(task, timeout=1)

    assert manager.is_interrupted is True

def test_reset_clears_interruption_signal():
    manager = VoiceTurnManager()
    manager.interrupt()

    manager.reset()

    assert manager.is_interrupted is False
    assert manager.state == VoiceState.IDLE

@pytest.mark.asyncio
async def test_run_interruptible_returns_operation_result():
    manager = VoiceTurnManager()

    async def operation():
        return "completed"

    result = await manager.run_interruptible(operation())

    assert result == "completed"


@pytest.mark.asyncio
async def test_run_interruptible_cancels_operation_on_interrupt():
    manager = VoiceTurnManager()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation():
        started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(
        manager.run_interruptible(operation())
    )

    await started.wait()
    manager.interrupt()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    assert manager.is_interrupted is True

@pytest.mark.asyncio
async def test_run_interruptible_cancels_operation_on_external_cancellation():
    manager = VoiceTurnManager()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation():
        started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(
        manager.run_interruptible(operation())
    )

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()

@pytest.mark.asyncio
async def test_run_interruptible_propagates_operation_exception():
    manager = VoiceTurnManager()

    async def operation():
        raise RuntimeError("Operation failed")

    with pytest.raises(RuntimeError, match="Operation failed"):
        await manager.run_interruptible(operation())

    assert manager.state == VoiceState.IDLE
    assert manager.is_interrupted is False