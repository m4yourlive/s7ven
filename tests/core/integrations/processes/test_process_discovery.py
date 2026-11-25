
"""Tests for psutil process discovery."""

from pathlib import Path
from unittest.mock import MagicMock

import psutil

from seven.core.integrations.processes.discovery import PsutilProcessDiscovery


def create_process(info: dict) -> MagicMock:
    process = MagicMock()
    process.info = info
    return process


def test_discovery_returns_process_snapshots(monkeypatch):
    processes = [
        create_process(
            {
                "pid": 200,
                "name": "python.exe",
                "exe": r"C:\Python\python.exe",
                "ppid": 100,
                "username": "user",
                "status": "running",
                "create_time": 1000.0,
                "memory_info": MagicMock(rss=2048),
            }
        ),
        create_process(
            {
                "pid": 100,
                "name": "explorer.exe",
                "exe": r"C:\Windows\explorer.exe",
                "ppid": 1,
                "username": "user",
                "status": "sleeping",
                "create_time": 900.0,
                "memory_info": MagicMock(rss=4096),
            }
        ),
    ]

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda **kwargs: iter(processes),
    )

    result = PsutilProcessDiscovery().discover()

    assert len(result) == 2
    assert result[0].pid == 100
    assert result[1].pid == 200
    assert result[1].name == "python.exe"
    assert result[1].executable == Path(r"C:\Python\python.exe")
    assert result[1].parent_pid == 100
    assert result[1].username == "user"
    assert result[1].status == "running"
    assert result[1].create_time == 1000.0
    assert result[1].memory_rss_bytes == 2048


def test_discovery_skips_processes_with_missing_required_fields(monkeypatch):
    processes = [
        create_process(
            {
                "pid": None,
                "name": "invalid.exe",
            }
        ),
        create_process(
            {
                "pid": 100,
                "name": "",
            }
        ),
        create_process(
            {
                "pid": 200,
                "name": "python.exe",
            }
        ),
    ]

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda **kwargs: iter(processes),
    )

    result = PsutilProcessDiscovery().discover()

    assert len(result) == 1
    assert result[0].pid == 200


def test_discovery_handles_missing_optional_fields(monkeypatch):
    process = create_process(
        {
            "pid": 123,
            "name": "python.exe",
            "exe": None,
            "ppid": None,
            "username": None,
            "status": None,
            "create_time": None,
            "memory_info": None,
        }
    )

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda **kwargs: iter([process]),
    )

    result = PsutilProcessDiscovery().discover()

    assert len(result) == 1
    assert result[0].executable is None
    assert result[0].parent_pid is None
    assert result[0].username is None
    assert result[0].status is None
    assert result[0].create_time is None
    assert result[0].memory_rss_bytes is None


def test_discovery_skips_invalid_process_snapshots(monkeypatch):
    processes = [
        create_process(
            {
                "pid": 100,
                "name": "python.exe",
                "ppid": True,
            }
        ),
        create_process(
            {
                "pid": 200,
                "name": "valid.exe",
                "ppid": 1,
            }
        ),
    ]

    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda **kwargs: iter(processes),
    )

    result = PsutilProcessDiscovery().discover()

    assert len(result) == 1
    assert result[0].pid == 200


def test_discovery_handles_process_iteration_errors(monkeypatch):
    def process_iterator(**kwargs):
        yield create_process(
            {
                "pid": 100,
                "name": "python.exe",
            }
        )
        raise psutil.AccessDenied(pid=200)

    monkeypatch.setattr(
        psutil,
        "process_iter",
        process_iterator,
    )

    result = PsutilProcessDiscovery().discover()

    assert len(result) == 1
    assert result[0].pid == 100


def test_discovery_returns_empty_tuple_when_no_processes_exist(monkeypatch):
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda **kwargs: iter([]),
    )

    result = PsutilProcessDiscovery().discover()

    assert result == ()
