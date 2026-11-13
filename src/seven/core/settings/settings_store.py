from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from seven.core.settings.exceptions import SettingsPersistenceError


class SettingsStore:
    """JSON-based persistent storage for Seven user settings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def exists(self) -> bool:
        """Return whether the settings file exists."""
        return self.path.is_file()

    def load(self) -> dict[str, Any]:
        """Load settings from a JSON file."""

        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            raise SettingsPersistenceError(
                f"Failed to load settings from {self.path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise SettingsPersistenceError(
                "Settings file root must be a JSON object."
            )

        return data

    def save(self, data: dict[str, Any]) -> None:
        """Atomically save settings to a JSON file."""

        if not isinstance(data, dict):
            raise TypeError("Settings data must be a dictionary.")

        temp_path: Path | None = None

        try:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)

                json.dump(
                    data,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(temp_path, self.path)
            temp_path = None

        except (OSError, TypeError, ValueError) as exc:
            raise SettingsPersistenceError(
                f"Failed to save settings to {self.path}: {exc}"
            ) from exc

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
