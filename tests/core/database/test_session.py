"""Tests for asynchronous database session management."""

import pytest
from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from seven.core.database import (
    Base,
    DatabaseConfig,
    DatabaseSessionManager,
)


class SessionTestModel(Base):
    """Test model for database session integration."""

    __tablename__ = "test_database_session_model"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)


@pytest.fixture
def database_config(tmp_path):
    """Provide an isolated database configuration."""
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="session_test.db",
    )


@pytest.fixture
async def session_manager(database_config):
    """Provide an initialized database session manager."""
    manager = DatabaseSessionManager(database_config)

    await manager.initialize()

    async with manager.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
        )

    yield manager

    await manager.dispose()


@pytest.mark.asyncio
async def test_manager_initially_not_initialized(
    database_config,
) -> None:
    manager = DatabaseSessionManager(database_config)

    try:
        assert manager.is_initialized is False
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_initialize_creates_database_directory(
    database_config,
) -> None:
    manager = DatabaseSessionManager(database_config)

    try:
        await manager.initialize()

        assert database_config.database_path.parent.exists()
        assert manager.is_initialized is True
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_initialize_is_idempotent(
    database_config,
) -> None:
    manager = DatabaseSessionManager(database_config)

    try:
        await manager.initialize()
        engine = manager.engine

        await manager.initialize()

        assert manager.engine is engine
        assert manager.is_initialized is True
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_session_requires_initialization(
    database_config,
) -> None:
    manager = DatabaseSessionManager(database_config)

    try:
        with pytest.raises(
            RuntimeError,
            match="has not been initialized",
        ):
            async with manager.session():
                pass
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_session_can_insert_and_read_data(
    session_manager,
) -> None:
    async with session_manager.session() as session:
        session.add(SessionTestModel(name="S7VEN"))
        await session.commit()

    async with session_manager.session() as session:
        result = await session.execute(
            select(SessionTestModel).where(
                SessionTestModel.name == "S7VEN"
            )
        )

        model = result.scalar_one()

        assert model.name == "S7VEN"


@pytest.mark.asyncio
async def test_session_does_not_automatically_commit(
    session_manager,
) -> None:
    async with session_manager.session() as session:
        session.add(SessionTestModel(name="Uncommitted"))

    async with session_manager.session() as session:
        result = await session.execute(
            select(SessionTestModel).where(
                SessionTestModel.name == "Uncommitted"
            )
        )

        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_session_rolls_back_on_context_exit(
    session_manager,
) -> None:
    async with session_manager.session() as session:
        session.add(SessionTestModel(name="Rollback Test"))
        await session.flush()

    async with session_manager.session() as session:
        result = await session.execute(
            select(SessionTestModel).where(
                SessionTestModel.name == "Rollback Test"
            )
        )

        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_dispose_resets_initialization_state(
    database_config,
) -> None:
    manager = DatabaseSessionManager(database_config)

    await manager.initialize()
    assert manager.is_initialized is True

    await manager.dispose()

    assert manager.is_initialized is False


@pytest.mark.asyncio
async def test_manager_can_reinitialize_after_dispose(
    database_config,
) -> None:
    manager = DatabaseSessionManager(database_config)

    try:
        await manager.initialize()
        await manager.dispose()

        await manager.initialize()

        assert manager.is_initialized is True

        async with manager.session() as session:
            assert session.is_active is True
    finally:
        await manager.dispose()
