
"""Tests for process inspection."""

from pathlib import Path
from unittest.mock import MagicMock

import psutil
import pytest

from seven.core.integrations.processes.inspection import (
    PsutilProcessInspector,
)


def create_process_mock():
    process = MagicMock()
    process.pid = 1234
    process.name.return_value = "python.exe"
    process.exe.return_value = r"C:\Python\python.exe"
    process.ppid.return_value = 100
    process.username.return_value = "user"
    process.status.return_value = "running"
    process.create_time.return_value = 1000.0
    process.cpu_percent.return_value = 12.5
    process.memory_info.return_value = MagicMock(rss=4096)
    process.oneshot.return_value.__enter__ = MagicMock()
    process.oneshot.return_value.__exit__ = MagicMock(return_value=False)
    return process


def test_inspection_returns_process_snapshot(monkeypatch):
    process = create_process_mock()

    monkeypatch.setattr(psutil, "Process", lambda pid: process)

    result = PsutilProcessInspector().inspect(1234)

    assert result.pid == 1234
    assert result.name == "python.exe"
    assert result.executable == Path(r"C:\Python\python.exe")
    assert result.parent_pid == 100
    assert result.username == "user"
    assert result.status == "running"
    assert result.create_time == 1000.0
    assert result.cpu_percent == 12.5
    assert result.memory_rss_bytes == 4096


@pytest.mark.parametrize("pid", [0, -1])
def test_inspection_rejects_invalid_pid(pid):
    with pytest.raises(ValueError):
        PsutilProcessInspector().inspect(pid)


@pytest.mark.parametrize("pid", [True, "1234", 1.5])
def test_inspection_rejects_non_integer_pid(pid):
    with pytest.raises(TypeError):
        PsutilProcessInspector().inspect(pid)


def test_inspection_propagates_no_such_process(monkeypatch):
    def raise_no_such_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", raise_no_such_process)

    with pytest.raises(psutil.NoSuchProcess):
        PsutilProcessInspector().inspect(1234)


def test_inspection_propagates_access_denied_on_process_creation(monkeypatch):
    def raise_access_denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", raise_access_denied)

    with pytest.raises(psutil.AccessDenied):
        PsutilProcessInspector().inspect(1234)


def test_inspection_handles_optional_access_denied(monkeypatch):
    process = create_process_mock()
    process.exe.side_effect = psutil.AccessDenied(1234)
    process.username.side_effect = psutil.AccessDenied(1234)
    process.status.side_effect = psutil.AccessDenied(1234)
    process.create_time.side_effect = psutil.AccessDenied(1234)
    process.memory_info.side_effect = psutil.AccessDenied(1234)
    process.cpu_percent.side_effect = psutil.AccessDenied(1234)

    monkeypatch.setattr(psutil, "Process", lambda pid: process)

    result = PsutilProcessInspector().inspect(1234)

    assert result.executable is None
    assert result.username is None
    assert result.status is None
    assert result.create_time is None
    assert result.cpu_percent is None
    assert result.memory_rss_bytes is None


def test_inspection_uses_oneshot_context(monkeypatch):
    process = create_process_mock()
    monkeypatch.setattr(psutil, "Process", lambda pid: process)

    PsutilProcessInspector().inspect(1234)

    process.oneshot.assert_called_once()
