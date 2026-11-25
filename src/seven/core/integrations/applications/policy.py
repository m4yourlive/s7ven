
from dataclasses import dataclass
from pathlib import Path

from seven.core.integrations.applications.models import Application


@dataclass(frozen=True, slots=True)
class ApplicationAccessPolicy:
    """Control which application executables may be launched."""

    allowed_executables: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[Path] = []

        for executable in self.allowed_executables:
            if not isinstance(executable, Path):
                raise TypeError(
                    "Allowed executables must be Path instances."
                )

            normalized.append(
                executable.expanduser().resolve()
            )

        object.__setattr__(
            self,
            "allowed_executables",
            tuple(dict.fromkeys(normalized)),
        )

    def authorize(self, application: Application) -> Path:
        """Authorize an application executable."""

        if not isinstance(application, Application):
            raise TypeError(
                "application must be an Application instance."
            )

        executable = application.executable.expanduser().resolve()

        if executable not in self.allowed_executables:
            raise PermissionError(
                f"Application executable is not allowed: {executable}"
            )

        if not executable.exists():
            raise FileNotFoundError(
                f"Application executable not found: {executable}"
            )

        if not executable.is_file():
            raise PermissionError(
                f"Application executable is not a file: {executable}"
            )

        return executable
