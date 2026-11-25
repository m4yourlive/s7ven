
"""Tests for Alembic database migrations."""

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from seven.core.database import DatabaseConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_CONFIG_PATH = PROJECT_ROOT / "alembic.ini"
EXPECTED_MIGRATION_REVISION = "c2a7e91f4b30"


@pytest.fixture
def database_config(tmp_path) -> DatabaseConfig:
    """Provide an isolated migration database configuration."""
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="migration_test.db",
    )


@pytest.fixture
def alembic_config(database_config) -> Config:
    """Provide an Alembic configuration for the test database."""
    config = Config(str(ALEMBIC_CONFIG_PATH))
    config.attributes["database_config"] = database_config

    return config


def get_revision(database_path: Path) -> str | None:
    """Read the current Alembic revision from SQLite."""
    with sqlite3.connect(database_path) as connection:
        result = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()

    return result[0] if result else None


def test_alembic_configuration_exists() -> None:
    assert ALEMBIC_CONFIG_PATH.exists()


def test_alembic_upgrade_creates_revision_table(
    alembic_config,
    database_config,
) -> None:
    command.upgrade(alembic_config, "head")

    assert database_config.database_path.exists()
    assert (
        get_revision(database_config.database_path)
        == EXPECTED_MIGRATION_REVISION
    )


def test_alembic_upgrade_is_idempotent(
    alembic_config,
    database_config,
) -> None:
    command.upgrade(alembic_config, "head")
    command.upgrade(alembic_config, "head")

    assert (
        get_revision(database_config.database_path)
        == EXPECTED_MIGRATION_REVISION
    )


def test_alembic_downgrade_removes_revision_tracking(
    alembic_config,
    database_config,
) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")

    with sqlite3.connect(database_config.database_path) as connection:
        result = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchall()

    assert result == []
