"""Database configuration and SQLite URL construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.engine import URL


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Store database configuration."""

    data_directory: Path = Path("data")
    database_name: str = "seven.db"

    def __post_init__(self) -> None:
        """Normalize and validate database configuration."""
        directory = Path(self.data_directory).expanduser()
        object.__setattr__(self, "data_directory", directory)

        if not isinstance(self.database_name, str):
            raise TypeError("Database name must be a string.")

        name = self.database_name.strip()

        if not name:
            raise ValueError("Database name cannot be empty.")

        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Database name must be a filename.")

        object.__setattr__(self, "database_name", name)

    @classmethod
    def from_storage_config(cls, storage_config: Any) -> DatabaseConfig:
        """Build database configuration from AppConfig storage settings."""
        return cls(
            data_directory=storage_config.data_directory,
            database_name=storage_config.database_name,
        )

    @property
    def database_path(self) -> Path:
        """Return the absolute database file path."""
        return (self.data_directory / self.database_name).resolve()

    @property
    def database_url(self) -> URL:
        """Return the SQLAlchemy asynchronous SQLite URL."""
        return URL.create(
            drivername="sqlite+aiosqlite",
            database=str(self.database_path),
        )
