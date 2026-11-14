"""Tests for database lifecycle and migration management."""

import sqlite3

import pytest
import pytest_asyncio

from seven.core.database import DatabaseConfig, DatabaseManager


@pytest.fixture
def database_config(tmp_path):
    """Provide an isolated database configuration."""
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="manager_test.db",
    )


@pytest_asyncio.fixture
async def database_manager(database_config):
    """Provide an initialized database manager."""
    manager = DatabaseManager(database_config)

    await manager.initialize()

    yield manager

    await manager.dispose()


@pytest.mark.asyncio
async def test_manager_initially_not_initialized(
    database_config,
) -> None:
    manager = DatabaseManager(database_config)

    try:
        assert manager.is_initialized is False
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_initialize_sets_initialized_state(
    database_config,
) -> None:
    manager = DatabaseManager(database_config)

    try:
        await manager.initialize()

        assert manager.is_initialized is True
        assert manager.session_manager.is_initialized is True
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_initialize_creates_database_file(
    database_config,
) -> None:
    manager = DatabaseManager(database_config)

    try:
        await manager.initialize()

        assert database_config.database_path.exists()
        assert database_config.database_path.is_file()
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_initialize_applies_alembic_migration(
    database_manager,
    database_config,
) -> None:
    with sqlite3.connect(database_config.database_path) as connection:
        result = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()

    assert result is not None
    assert result[0] == "0001_initial"


@pytest.mark.asyncio
async def test_initialize_is_idempotent(
    database_manager,
) -> None:
    engine = database_manager.engine

    await database_manager.initialize()

    assert database_manager.engine is engine
    assert database_manager.is_initialized is True


@pytest.mark.asyncio
async def test_database_manager_exposes_configuration(
    database_config,
) -> None:
    manager = DatabaseManager(database_config)

    try:
        assert manager.config is database_config
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_dispose_resets_manager_state(
    database_manager,
) -> None:
    await database_manager.dispose()

    assert database_manager.is_initialized is False
    assert database_manager.session_manager.is_initialized is False


@pytest.mark.asyncio
async def test_manager_can_reinitialize_after_dispose(
    database_manager,
) -> None:
    await database_manager.dispose()

    await database_manager.initialize()

    assert database_manager.is_initialized is True
    assert database_manager.session_manager.is_initialized is True

    with sqlite3.connect(
        database_manager.config.database_path
    ) as connection:
        result = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()

    assert result[0] == "0001_initial"
