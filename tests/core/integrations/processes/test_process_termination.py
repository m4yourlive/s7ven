
"""Tests for process termination."""

from unittest.mock import MagicMock

import psutil
import pytest

from seven.core.integrations.processes.termination import (
    PsutilProcessTerminator,
)


def test_termination_sends_graceful_termination(monkeypatch):
    process = MagicMock()
    monkeypatch.setattr(psutil, "Process", lambda pid: process)

    PsutilProcessTerminator().terminate(1234)

    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=3.0)
    process.kill.assert_not_called()


def test_termination_force_kills_after_timeout(monkeypatch):
    process = MagicMock()
    process.wait.side_effect = [
        psutil.TimeoutExpired(3.0, pid=1234),
        0,
    ]
    monkeypatch.setattr(psutil, "Process", lambda pid: process)

    PsutilProcessTerminator(timeout=3.0).terminate(1234)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_termination_rejects_invalid_pid():
    terminator = PsutilProcessTerminator()

    with pytest.raises(ValueError):
        terminator.terminate(0)

    with pytest.raises(ValueError):
        terminator.terminate(-1)


def test_termination_rejects_non_integer_pid():
    terminator = PsutilProcessTerminator()

    with pytest.raises(TypeError):
        terminator.terminate(True)

    with pytest.raises(TypeError):
        terminator.terminate("1234")


def test_termination_rejects_protected_process(monkeypatch):
    process_factory = MagicMock()
    monkeypatch.setattr(psutil, "Process", process_factory)

    terminator = PsutilProcessTerminator(
        protected_pids=frozenset({1234}),
    )

    with pytest.raises(PermissionError):
        terminator.terminate(1234)

    process_factory.assert_not_called()


def test_termination_propagates_process_not_found(monkeypatch):
    def raise_no_such_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", raise_no_such_process)

    with pytest.raises(psutil.NoSuchProcess):
        PsutilProcessTerminator().terminate(1234)


def test_termination_propagates_access_denied(monkeypatch):
    process = MagicMock()
    process.terminate.side_effect = psutil.AccessDenied(1234)
    monkeypatch.setattr(psutil, "Process", lambda pid: process)

    with pytest.raises(psutil.AccessDenied):
        PsutilProcessTerminator().terminate(1234)

    process.kill.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1])
def test_termination_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        PsutilProcessTerminator(timeout=timeout)


def test_termination_rejects_non_numeric_timeout():
    with pytest.raises(TypeError):
        PsutilProcessTerminator(timeout="3")


def test_termination_rejects_invalid_protected_pid():
    with pytest.raises(TypeError):
        PsutilProcessTerminator(
            protected_pids=frozenset({True}),
        )
