
from pathlib import Path
from collections.abc import Iterable

from seven.core.integrations.applications.models import Application


class DirectoryApplicationDiscovery:
    """Discover applications from configured executable directories."""

    def __init__(
        self,
        directories: Iterable[Path],
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        self._directories = tuple(
            Path(directory).expanduser()
            for directory in directories
        )

        self._extensions = extensions or self._default_extensions()

        normalized_extensions = tuple(
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in self._extensions
        )

        if not normalized_extensions:
            raise ValueError("At least one executable extension is required.")

        self._extensions = normalized_extensions

    @staticmethod
    def _default_extensions() -> tuple[str, ...]:
        import os

        if os.name == "nt":
            return (".exe", ".bat", ".cmd")

        if os.name == "posix":
            return ("",)

        return ()

    def discover(self) -> tuple[Application, ...]:
        """Return executable files found in configured directories."""

        discovered: dict[str, Application] = {}

        for directory in self._directories:
            if not directory.exists() or not directory.is_dir():
                continue

            try:
                entries = sorted(
                    directory.iterdir(),
                    key=lambda path: path.name.casefold(),
                )
            except OSError:
                continue

            for entry in entries:
                try:
                    if entry.is_symlink() or not entry.is_file():
                        continue

                    if not self._matches_extension(entry):
                        continue

                    if not self._is_executable(entry):
                        continue

                    resolved = entry.resolve()
                    key = str(resolved).casefold()

                    discovered.setdefault(
                        key,
                        Application(
                            name=entry.stem,
                            executable=resolved,
                        ),
                    )
                except OSError:
                    continue

        return tuple(
            discovered[key]
            for key in sorted(discovered)
        )

    def _matches_extension(self, path: Path) -> bool:
        if "" in self._extensions:
            return True

        return path.suffix.lower() in self._extensions

    def _is_executable(self, path: Path) -> bool:
        import os

        if os.name == "nt":
            return path.suffix.lower() in self._extensions

        return os.access(path, os.X_OK)
