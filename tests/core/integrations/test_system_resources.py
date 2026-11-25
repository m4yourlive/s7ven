
from seven.core.integrations.system import (
    CPUInformation,
    DiskInformation,
    MemoryInformation,
    SystemResourceSnapshot,
    detect_cpu_information,
    detect_disk_information,
    detect_memory_information,
    detect_system_resource_snapshot,
)


def test_detect_cpu_information_returns_valid_result() -> None:
    result = detect_cpu_information(interval=0)

    assert isinstance(result, CPUInformation)
    assert 0 <= result.usage_percent <= 100
    assert result.logical_cores > 0
    assert result.physical_cores is None or result.physical_cores > 0


def test_detect_memory_information_returns_valid_result() -> None:
    result = detect_memory_information()

    assert isinstance(result, MemoryInformation)
    assert result.total_bytes > 0
    assert result.available_bytes >= 0
    assert result.used_bytes >= 0
    assert 0 <= result.usage_percent <= 100


def test_detect_disk_information_returns_tuple() -> None:
    result = detect_disk_information()

    assert isinstance(result, tuple)

    for disk in result:
        assert isinstance(disk, DiskInformation)
        assert disk.total_bytes > 0
        assert disk.used_bytes >= 0
        assert disk.free_bytes >= 0
        assert 0 <= disk.usage_percent <= 100


def test_detect_system_resource_snapshot_returns_valid_result() -> None:
    result = detect_system_resource_snapshot(cpu_interval=0)

    assert isinstance(result, SystemResourceSnapshot)
    assert isinstance(result.cpu, CPUInformation)
    assert isinstance(result.memory, MemoryInformation)
    assert isinstance(result.disks, tuple)
    assert result.uptime_seconds >= 0
    assert result.captured_at > 0


def test_resource_models_are_immutable() -> None:
    result = detect_memory_information()

    try:
        result.total_bytes = 0
    except AttributeError:
        pass
    else:
        raise AssertionError("MemoryInformation must be immutable.")
