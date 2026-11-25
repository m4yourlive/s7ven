
"""Tests for process domain models."""

from pathlib import Path

import pytest

from seven.core.integrations.processes.models import ProcessInfo


def test_process_info_accepts_valid_data():
    process = ProcessInfo(
        pid=1234,
        name="  python  ",
        executable=Path("~/python.exe"),
        parent_pid=100,
        username="user",
        status="running",
        create_time=1000.5,
        cpu_percent=12.5,
        memory_rss_bytes=1024,
    )

    assert process.pid == 1234
    assert process.name == "python"
    assert process.executable == Path("~/python.exe").expanduser()
    assert process.parent_pid == 100
    assert process.username == "user"
    assert process.status == "running"
    assert process.create_time == 1000.5
    assert process.cpu_percent == 12.5
    assert process.memory_rss_bytes == 1024


def test_process_info_accepts_optional_fields():
    process = ProcessInfo(pid=1234, name="python")

    assert process.executable is None
    assert process.parent_pid is None
    assert process.username is None
    assert process.status is None
    assert process.create_time is None
    assert process.cpu_percent is None
    assert process.memory_rss_bytes is None


@pytest.mark.parametrize("pid", [0, -1])
def test_process_info_rejects_invalid_pid(pid):
    with pytest.raises(ValueError):
        ProcessInfo(pid=pid, name="python")


@pytest.mark.parametrize("pid", [True, "1234", 1.5])
def test_process_info_rejects_non_integer_pid(pid):
    with pytest.raises(TypeError):
        ProcessInfo(pid=pid, name="python")


@pytest.mark.parametrize("name", ["", "   "])
def test_process_info_rejects_empty_name(name):
    with pytest.raises(ValueError):
        ProcessInfo(pid=1234, name=name)


def test_process_info_rejects_non_string_name():
    with pytest.raises(TypeError):
        ProcessInfo(pid=1234, name=123)


def test_process_info_rejects_invalid_executable():
    with pytest.raises(TypeError):
        ProcessInfo(
            pid=1234,
            name="python",
            executable="python.exe",
        )


@pytest.mark.parametrize("parent_pid", [0, -1])
def test_process_info_rejects_invalid_parent_pid(parent_pid):
    with pytest.raises(ValueError):
        ProcessInfo(
            pid=1234,
            name="python",
            parent_pid=parent_pid,
        )


def test_process_info_rejects_non_integer_parent_pid():
    with pytest.raises(TypeError):
        ProcessInfo(
            pid=1234,
            name="python",
            parent_pid=True,
        )


@pytest.mark.parametrize("field_name", ["username", "status"])
def test_process_info_rejects_invalid_optional_strings(field_name):
    with pytest.raises(TypeError):
        ProcessInfo(
            pid=1234,
            name="python",
            **{field_name: 123},
        )


@pytest.mark.parametrize(
    "create_time",
    [-1, float("inf"), float("nan")],
)
def test_process_info_rejects_invalid_creation_time(create_time):
    with pytest.raises(ValueError):
        ProcessInfo(
            pid=1234,
            name="python",
            create_time=create_time,
        )


@pytest.mark.parametrize(
    "cpu_percent",
    [-1, float("inf"), float("nan")],
)
def test_process_info_rejects_invalid_cpu_percentage(cpu_percent):
    with pytest.raises(ValueError):
        ProcessInfo(
            pid=1234,
            name="python",
            cpu_percent=cpu_percent,
        )


@pytest.mark.parametrize("memory", [-1, 1.5, True])
def test_process_info_rejects_invalid_memory_usage(memory):
    expected_error = TypeError if isinstance(memory, (float, bool)) else ValueError

    with pytest.raises(expected_error):
        ProcessInfo(
            pid=1234,
            name="python",
            memory_rss_bytes=memory,
        )


def test_process_info_is_immutable():
    process = ProcessInfo(pid=1234, name="python")

    with pytest.raises(AttributeError):
        process.pid = 5678
