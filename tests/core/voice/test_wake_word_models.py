"""Tests for wake word data models."""

import pytest

from seven.core.voice.models import WakeWordResult


def test_creates_detected_result():
    result = WakeWordResult(
        detected=True,
        wake_word="Hey Seven",
    )

    assert result.detected is True
    assert result.wake_word == "Hey Seven"


def test_creates_not_detected_result():
    result = WakeWordResult(detected=False)

    assert result.detected is False
    assert result.wake_word is None


def test_rejects_detected_result_without_wake_word():
    with pytest.raises(
        ValueError,
        match="Wake word must be provided",
    ):
        WakeWordResult(detected=True)