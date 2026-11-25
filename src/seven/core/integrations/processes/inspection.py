
"""Process inspection implementation using psutil."""

import psutil

from seven.core.integrations.processes.models import ProcessInfo


class PsutilProcessInspector:
    """Inspect individual operating system processes."""

    def inspect(self, pid: int) -> ProcessInfo:
        """Return a snapshot of the requested process."""
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise TypeError("Process ID must be an integer.")

        if pid <= 0:
            raise ValueError("Process ID must be greater than zero.")

        process = psutil.Process(pid)

        with process.oneshot():
            try:
                memory_info = process.memory_info()
            except psutil.AccessDenied:
                memory_info = None

            try:
                cpu_percent = process.cpu_percent(interval=None)
            except psutil.AccessDenied:
                cpu_percent = None

            try:
                executable = process.exe()
            except psutil.AccessDenied:
                executable = None

            return ProcessInfo(
                pid=process.pid,
                name=process.name(),
                executable=(
                    __import__("pathlib").Path(executable)
                    if executable
                    else None
                ),
                parent_pid=process.ppid(),
                username=self._safe_read(process.username),
                status=self._safe_read(process.status),
                create_time=self._safe_read(process.create_time),
                cpu_percent=cpu_percent,
                memory_rss_bytes=(
                    memory_info.rss if memory_info is not None else None
                ),
            )

    @staticmethod
    def _safe_read(reader):
        """Read an optional process attribute safely."""
        try:
            return reader()
        except psutil.AccessDenied:
            return None
