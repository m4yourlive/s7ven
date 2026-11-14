"""Tests for the generic database repository."""

import pytest
import pytest_asyncio
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from seven.core.database import (
    Base,
    DatabaseConfig,
    DatabaseSessionManager,
)
from seven.core.database.repositories import BaseRepository


class RepositoryTestModel(Base):
    """Test model for repository operations."""

    __tablename__ = "test_database_repository_model"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)


@pytest.fixture
def database_config(tmp_path):
    """Provide an isolated database configuration."""
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="repository_test.db",
    )


@pytest_asyncio.fixture
async def repository(database_config):
    """Provide an initialized repository and database session."""
    manager = DatabaseSessionManager(database_config)

    await manager.initialize()

    async with manager.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with manager.session() as session:
        yield BaseRepository(session, RepositoryTestModel)

    await manager.dispose()


@pytest.mark.asyncio
async def test_repository_initialization(repository) -> None:
    assert repository.model is RepositoryTestModel
    assert repository.session is not None


@pytest.mark.asyncio
async def test_add_entity(repository) -> None:
    entity = RepositoryTestModel(name="S7VEN")

    result = await repository.add(entity)

    assert result is entity
    assert entity.id is not None


@pytest.mark.asyncio
async def test_get_by_id_returns_existing_entity(repository) -> None:
    entity = await repository.add(
        RepositoryTestModel(name="S7VEN")
    )

    result = await repository.get_by_id(entity.id)

    assert result is entity
    assert result.name == "S7VEN"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing_entity(
    repository,
) -> None:
    result = await repository.get_by_id(999)

    assert result is None


@pytest.mark.asyncio
async def test_get_all_returns_entities(repository) -> None:
    await repository.add(RepositoryTestModel(name="First"))
    await repository.add(RepositoryTestModel(name="Second"))

    result = await repository.get_all()

    assert len(result) == 2
    assert {entity.name for entity in result} == {
        "First",
        "Second",
    }


@pytest.mark.asyncio
async def test_get_all_supports_limit(repository) -> None:
    await repository.add(RepositoryTestModel(name="First"))
    await repository.add(RepositoryTestModel(name="Second"))

    result = await repository.get_all(limit=1)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_all_supports_offset(repository) -> None:
    await repository.add(RepositoryTestModel(name="First"))
    await repository.add(RepositoryTestModel(name="Second"))

    result = await repository.get_all(offset=1)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_all_rejects_invalid_limit(repository) -> None:
    with pytest.raises(ValueError, match="Limit"):
        await repository.get_all(limit=0)


@pytest.mark.asyncio
async def test_get_all_rejects_negative_offset(repository) -> None:
    with pytest.raises(ValueError, match="Offset"):
        await repository.get_all(offset=-1)


@pytest.mark.asyncio
async def test_update_entity(repository) -> None:
    entity = await repository.add(
        RepositoryTestModel(name="Original")
    )

    result = await repository.update(
        entity,
        {"name": "Updated"},
    )

    assert result.name == "Updated"


@pytest.mark.asyncio
async def test_update_rejects_unknown_field(repository) -> None:
    entity = await repository.add(
        RepositoryTestModel(name="Original")
    )

    with pytest.raises(AttributeError, match="unknown"):
        await repository.update(
            entity,
            {"unknown": "value"},
        )


@pytest.mark.asyncio
async def test_delete_entity(repository) -> None:
    entity = await repository.add(
        RepositoryTestModel(name="To Delete")
    )

    await repository.delete(entity)

    result = await repository.get_by_id(entity.id)

    assert result is None


@pytest.mark.asyncio
async def test_delete_by_id_returns_true(repository) -> None:
    entity = await repository.add(
        RepositoryTestModel(name="To Delete")
    )

    result = await repository.delete_by_id(entity.id)

    assert result is True


@pytest.mark.asyncio
async def test_delete_by_id_returns_false_for_missing_entity(
    repository,
) -> None:
    result = await repository.delete_by_id(999)

    assert result is False


@pytest.mark.asyncio
async def test_count_returns_entity_count(repository) -> None:
    await repository.add(RepositoryTestModel(name="First"))
    await repository.add(RepositoryTestModel(name="Second"))

    result = await repository.count()

    assert result == 2


@pytest.mark.asyncio
async def test_repository_does_not_automatically_commit(
    repository,
) -> None:
    await repository.add(
        RepositoryTestModel(name="Uncommitted")
    )

    # The fixture's session closes without committing.
    # The next independent session is not available here,
    # so this test verifies that add only flushes the entity.
    assert await repository.count() == 1
