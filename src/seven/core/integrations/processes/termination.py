
"""Process termination implementation using psutil."""

import psutil


class PsutilProcessTerminator:
    """Terminate operating system processes safely."""

    def __init__(
        self,
        timeout: float = 3.0,
        protected_pids: frozenset[int] | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("Timeout must be numeric.")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self._timeout = float(timeout)
        self._protected_pids = protected_pids or frozenset()

        for pid in self._protected_pids:
            if isinstance(pid, bool) or not isinstance(pid, int):
                raise TypeError("Protected process IDs must be integers.")

            if pid <= 0:
                raise ValueError(
                    "Protected process IDs must be greater than zero."
                )

    def terminate(self, pid: int) -> None:
        """Request graceful termination, then force kill after timeout."""
        self._validate_pid(pid)

        if pid in self._protected_pids:
            raise PermissionError(
                f"Process {pid} is protected and cannot be terminated."
            )

        process = psutil.Process(pid)

        try:
            process.terminate()
            process.wait(timeout=self._timeout)

        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._timeout)

    @staticmethod
    def _validate_pid(pid: int) -> None:
        """Validate a process identifier."""
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise TypeError("Process ID must be an integer.")

        if pid <= 0:
            raise ValueError("Process ID must be greater than zero.")
