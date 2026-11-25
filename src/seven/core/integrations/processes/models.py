
"""Domain models for operating system processes."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """Represent a snapshot of an operating system process."""

    pid: int
    name: str
    executable: Path | None = None
    parent_pid: int | None = None
    username: str | None = None
    status: str | None = None
    create_time: float | None = None
    cpu_percent: float | None = None
    memory_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int):
            raise TypeError("Process ID must be an integer.")

        if self.pid <= 0:
            raise ValueError("Process ID must be greater than zero.")

        if not isinstance(self.name, str):
            raise TypeError("Process name must be a string.")

        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Process name cannot be empty.")

        object.__setattr__(self, "name", normalized_name)

        if self.executable is not None:
            if not isinstance(self.executable, Path):
                raise TypeError("Executable must be a Path or None.")

            object.__setattr__(
                self,
                "executable",
                self.executable.expanduser(),
            )

        if self.parent_pid is not None:
            if (
                isinstance(self.parent_pid, bool)
                or not isinstance(self.parent_pid, int)
            ):
                raise TypeError("Parent process ID must be an integer.")

            if self.parent_pid <= 0:
                raise ValueError(
                    "Parent process ID must be greater than zero."
                )

        for field_name in ("username", "status"):
            value = getattr(self, field_name)

            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None.")

        if self.create_time is not None:
            if (
                isinstance(self.create_time, bool)
                or not isinstance(self.create_time, (int, float))
            ):
                raise TypeError("Creation time must be numeric.")

            if not isfinite(self.create_time) or self.create_time < 0:
                raise ValueError(
                    "Creation time must be finite and non-negative."
                )

        if self.cpu_percent is not None:
            if (
                isinstance(self.cpu_percent, bool)
                or not isinstance(self.cpu_percent, (int, float))
            ):
                raise TypeError("CPU percentage must be numeric.")

            if not isfinite(self.cpu_percent) or self.cpu_percent < 0:
                raise ValueError(
                    "CPU percentage must be finite and non-negative."
                )

        if self.memory_rss_bytes is not None:
            if (
                isinstance(self.memory_rss_bytes, bool)
                or not isinstance(self.memory_rss_bytes, int)
            ):
                raise TypeError("Memory usage must be an integer.")

            if self.memory_rss_bytes < 0:
                raise ValueError("Memory usage cannot be negative.")
