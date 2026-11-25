
"""Access policy for operating system processes."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessAccessPolicy:
    """Define which processes may be inspected or terminated."""

    allow_discovery: bool = True
    allow_inspection: bool = True
    allow_termination: bool = False
    protected_pids: frozenset[int] = frozenset()
    allowed_termination_pids: frozenset[int] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "allow_discovery",
            "allow_inspection",
            "allow_termination",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")

        self._validate_pid_collection(
            self.protected_pids,
            "protected_pids",
        )

        if self.allowed_termination_pids is not None:
            self._validate_pid_collection(
                self.allowed_termination_pids,
                "allowed_termination_pids",
            )

    def authorize_discovery(self) -> None:
        """Authorize process discovery."""
        if not self.allow_discovery:
            raise PermissionError("Process discovery is not permitted.")

    def authorize_inspection(self, pid: int) -> None:
        """Authorize inspection of a process."""
        self._validate_pid(pid)

        if not self.allow_inspection:
            raise PermissionError("Process inspection is not permitted.")

    def authorize_termination(self, pid: int) -> None:
        """Authorize termination of a process."""
        self._validate_pid(pid)

        if not self.allow_termination:
            raise PermissionError("Process termination is not permitted.")

        if pid in self.protected_pids:
            raise PermissionError(
                f"Process {pid} is protected and cannot be terminated."
            )

        if (
            self.allowed_termination_pids is not None
            and pid not in self.allowed_termination_pids
        ):
            raise PermissionError(
                f"Process {pid} is not in the termination allowlist."
            )

    @staticmethod
    def _validate_pid(pid: int) -> None:
        """Validate a process identifier."""
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise TypeError("Process ID must be an integer.")

        if pid <= 0:
            raise ValueError("Process ID must be greater than zero.")

    @classmethod
    def _validate_pid_collection(
        cls,
        pids: frozenset[int],
        field_name: str,
    ) -> None:
        """Validate a collection of process identifiers."""
        if not isinstance(pids, frozenset):
            raise TypeError(f"{field_name} must be a frozenset.")

        for pid in pids:
            cls._validate_pid(pid)
