
"""Process discovery implementation using psutil."""

from pathlib import Path

import psutil

from seven.core.integrations.processes.models import ProcessInfo


class PsutilProcessDiscovery:
    """Discover running processes through psutil."""

    def discover(self) -> tuple[ProcessInfo, ...]:
        """Return available process snapshots in PID order."""
        processes: list[ProcessInfo] = []

        try:
            iterator = psutil.process_iter(
                attrs=[
                    "pid",
                    "name",
                    "exe",
                    "ppid",
                    "username",
                    "status",
                    "create_time",
                    "memory_info",
                ],
                ad_value=None,
            )

            while True:
                try:
                    process = next(iterator)
                except StopIteration:
                    break
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue

                try:
                    info = process.info
                    pid = info.get("pid")
                    name = info.get("name")

                    if (
                        isinstance(pid, bool)
                        or not isinstance(pid, int)
                        or pid <= 0
                    ):
                        continue

                    if not isinstance(name, str) or not name.strip():
                        continue

                    memory_info = info.get("memory_info")
                    memory_rss = (
                        memory_info.rss
                        if memory_info is not None
                        else None
                    )

                    executable = info.get("exe")

                    processes.append(
                        ProcessInfo(
                            pid=pid,
                            name=name,
                            executable=(
                                Path(executable)
                                if executable
                                else None
                            ),
                            parent_pid=info.get("ppid"),
                            username=info.get("username"),
                            status=info.get("status"),
                            create_time=info.get("create_time"),
                            memory_rss_bytes=memory_rss,
                        )
                    )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                    TypeError,
                    ValueError,
                ):
                    continue

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            pass

        return tuple(sorted(processes, key=lambda item: item.pid))
