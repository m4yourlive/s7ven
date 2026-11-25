
import time
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class CPUInformation:
    usage_percent: float
    logical_cores: int
    physical_cores: int | None
    frequency_mhz: float | None


@dataclass(frozen=True, slots=True)
class MemoryInformation:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    usage_percent: float


@dataclass(frozen=True, slots=True)
class DiskInformation:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float


@dataclass(frozen=True, slots=True)
class SystemResourceSnapshot:
    cpu: CPUInformation
    memory: MemoryInformation
    disks: tuple[DiskInformation, ...]
    uptime_seconds: float
    captured_at: float


def detect_cpu_information(
    interval: float = 0.1,
) -> CPUInformation:
    if interval < 0:
        raise ValueError("CPU sampling interval cannot be negative.")

    frequency = psutil.cpu_freq()

    return CPUInformation(
        usage_percent=psutil.cpu_percent(interval=interval),
        logical_cores=psutil.cpu_count(logical=True) or 0,
        physical_cores=psutil.cpu_count(logical=False),
        frequency_mhz=(
            frequency.current if frequency is not None else None
        ),
    )


def detect_memory_information() -> MemoryInformation:
    memory = psutil.virtual_memory()

    return MemoryInformation(
        total_bytes=memory.total,
        available_bytes=memory.available,
        used_bytes=memory.used,
        usage_percent=memory.percent,
    )


def detect_disk_information() -> tuple[DiskInformation, ...]:
    disks: list[DiskInformation] = []

    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except (PermissionError, OSError):
            continue

        disks.append(
            DiskInformation(
                path=partition.mountpoint,
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                usage_percent=usage.percent,
            )
        )

    return tuple(disks)


def detect_system_resource_snapshot(
    cpu_interval: float = 0.1,
) -> SystemResourceSnapshot:
    boot_time = psutil.boot_time()

    return SystemResourceSnapshot(
        cpu=detect_cpu_information(interval=cpu_interval),
        memory=detect_memory_information(),
        disks=detect_disk_information(),
        uptime_seconds=max(0.0, time.time() - boot_time),
        captured_at=time.time(),
    )
