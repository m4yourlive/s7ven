"""End-to-end database integration tests."""

import pytest
import pytest_asyncio
from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from seven.core.database import Base, DatabaseConfig, DatabaseManager
from seven.core.database.repositories import BaseRepository


class IntegrationTestModel(Base):
    """Model used for database integration tests."""

    __tablename__ = "test_database_integration_model"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)


@pytest.fixture
def database_config(tmp_path):
    """Provide an isolated database configuration."""
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="integration_test.db",
    )


@pytest_asyncio.fixture
async def database_manager(database_config):
    """Provide an initialized database manager."""
    manager = DatabaseManager(database_config)

    await manager.initialize()

    async with manager.engine.begin() as connection:
        await connection.run_sync(
            lambda sync_connection: Base.metadata.create_all(
                sync_connection,
                tables=[IntegrationTestModel.__table__],
            )
        )

    yield manager

    await manager.dispose()


@pytest.mark.asyncio
async def test_repository_data_persists_after_commit(
    database_manager,
) -> None:
    async with database_manager.session_manager.session() as session:
        repository = BaseRepository(
            session,
            IntegrationTestModel,
        )

        entity = await repository.add(
            IntegrationTestModel(name="S7VEN")
        )

        entity_id = entity.id
        await session.commit()

    async with database_manager.session_manager.session() as session:
        repository = BaseRepository(
            session,
            IntegrationTestModel,
        )

        result = await repository.get_by_id(entity_id)

        assert result is not None
        assert result.name == "S7VEN"


@pytest.mark.asyncio
async def test_uncommitted_data_is_rolled_back(
    database_manager,
) -> None:
    async with database_manager.session_manager.session() as session:
        repository = BaseRepository(
            session,
            IntegrationTestModel,
        )

        await repository.add(
            IntegrationTestModel(name="Uncommitted")
        )

    async with database_manager.session_manager.session() as session:
        result = await session.execute(
            select(IntegrationTestModel).where(
                IntegrationTestModel.name == "Uncommitted"
            )
        )

        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_repository_update_and_delete(
    database_manager,
) -> None:
    async with database_manager.session_manager.session() as session:
        repository = BaseRepository(
            session,
            IntegrationTestModel,
        )

        entity = await repository.add(
            IntegrationTestModel(name="Original")
        )

        await repository.update(
            entity,
            {"name": "Updated"},
        )

        await session.commit()

        entity_id = entity.id

    async with database_manager.session_manager.session() as session:
        repository = BaseRepository(
            session,
            IntegrationTestModel,
        )

        entity = await repository.get_by_id(entity_id)

        assert entity is not None
        assert entity.name == "Updated"

        await repository.delete(entity)
        await session.commit()

        assert await repository.get_by_id(entity_id) is None
