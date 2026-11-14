"""Tests for database configuration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from seven.core.database import DatabaseConfig


def test_database_config_uses_default_values() -> None:
    config = DatabaseConfig()

    assert config.data_directory == Path("data")
    assert config.database_name == "seven.db"


def test_database_config_accepts_custom_values(
    tmp_path: Path,
) -> None:
    config = DatabaseConfig(
        data_directory=tmp_path,
        database_name="custom.db",
    )

    assert config.data_directory == tmp_path
    assert config.database_name == "custom.db"


def test_database_config_normalizes_data_directory(
    tmp_path: Path,
) -> None:
    directory = str(tmp_path / "database")

    config = DatabaseConfig(data_directory=Path(directory))

    assert config.data_directory == Path(directory)


def test_database_path_is_absolute(
    tmp_path: Path,
) -> None:
    config = DatabaseConfig(
        data_directory=tmp_path,
        database_name="seven.db",
    )

    assert config.database_path.is_absolute()
    assert config.database_path == tmp_path / "seven.db"


def test_database_url_uses_async_sqlite_driver(
    tmp_path: Path,
) -> None:
    config = DatabaseConfig(
        data_directory=tmp_path,
        database_name="seven.db",
    )

    assert config.database_url.drivername == "sqlite+aiosqlite"
    assert config.database_url.database == str(
        tmp_path / "seven.db"
    )


def test_database_config_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        DatabaseConfig(database_name="   ")


def test_database_config_rejects_path_separator() -> None:
    with pytest.raises(ValueError, match="must be a filename"):
        DatabaseConfig(database_name="folder/seven.db")


def test_database_config_rejects_windows_path_separator() -> None:
    with pytest.raises(ValueError, match="must be a filename"):
        DatabaseConfig(database_name="folder\\seven.db")


def test_database_config_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        DatabaseConfig(database_name=123)  # type: ignore[arg-type]


def test_database_config_from_storage_config(
    tmp_path: Path,
) -> None:
    storage_config = SimpleNamespace(
        data_directory=tmp_path,
        database_name="seven.db",
    )

    config = DatabaseConfig.from_storage_config(storage_config)

    assert config.data_directory == tmp_path
    assert config.database_name == "seven.db"


def test_database_config_is_immutable(
    tmp_path: Path,
) -> None:
    config = DatabaseConfig(data_directory=tmp_path)

    with pytest.raises(AttributeError):
        config.database_name = "changed.db"  # type: ignore[misc]


def test_database_config_does_not_create_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "not_created"

    DatabaseConfig(data_directory=directory)

    assert not directory.exists()
