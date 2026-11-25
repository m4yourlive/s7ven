
import subprocess
from pathlib import Path

from seven.core.integrations.applications.models import Application


class SubprocessApplicationLauncher:
    """Launch desktop applications without invoking a shell."""

    def __init__(
        self,
        working_directory: Path | None = None,
    ) -> None:
        self._working_directory = (
            working_directory.expanduser().resolve()
            if working_directory is not None
            else None
        )

    def launch(self, application: Application) -> int:
        """Launch an application and return its process identifier."""

        if not isinstance(application, Application):
            raise TypeError("application must be an Application instance.")

        executable = application.executable.expanduser().resolve()

        if not executable.exists():
            raise FileNotFoundError(
                f"Application executable not found: {executable}"
            )

        if not executable.is_file():
            raise ValueError(
                f"Application executable is not a file: {executable}"
            )

        command = [
            str(executable),
            *application.arguments,
        ]

        process = subprocess.Popen(
            command,
            cwd=(
                str(self._working_directory)
                if self._working_directory is not None
                else None
            ),
            shell=False,
            close_fds=True,
        )

        return process.pid
