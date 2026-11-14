"""Generic asynchronous SQLAlchemy repository."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.database.base import Base


class BaseRepository[ModelT: Base]:
    """Provide reusable CRUD operations for SQLAlchemy models."""

    def __init__(
        self,
        session: AsyncSession,
        model: type[ModelT],
    ) -> None:
        """Initialize the repository with a session and model."""
        self._session = session
        self._model = model

    @property
    def session(self) -> AsyncSession:
        """Return the managed database session."""
        return self._session

    @property
    def model(self) -> type[ModelT]:
        """Return the repository model."""
        return self._model

    async def get_by_id(self, entity_id: Any) -> ModelT | None:
        """Retrieve an entity by its primary key."""
        return await self._session.get(self._model, entity_id)

    async def get_all(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ModelT]:
        """Retrieve entities with optional pagination."""
        if limit is not None and limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        statement = select(self._model).offset(offset)

        if limit is not None:
            statement = statement.limit(limit)

        result = await self._session.execute(statement)

        return list(result.scalars().all())

    async def add(self, entity: ModelT) -> ModelT:
        """Add an entity to the current session."""
        self._session.add(entity)
        await self._session.flush()

        return entity

    async def update(
        self,
        entity: ModelT,
        values: dict[str, Any],
    ) -> ModelT:
        """Update an entity using the supplied field values."""
        for field, value in values.items():
            if not hasattr(entity, field):
                raise AttributeError(
                    f"{self._model.__name__} has no attribute '{field}'."
                )

            setattr(entity, field, value)

        await self._session.flush()

        return entity

    async def delete(self, entity: ModelT) -> None:
        """Delete an entity from the current session."""
        await self._session.delete(entity)
        await self._session.flush()

    async def delete_by_id(self, entity_id: Any) -> bool:
        """Delete an entity by its primary key."""
        entity = await self.get_by_id(entity_id)

        if entity is None:
            return False

        await self.delete(entity)

        return True

    async def count(self) -> int:
        """Return the total number of entities."""
        statement = select(func.count()).select_from(self._model)

        result = await self._session.execute(statement)

        return result.scalar_one()
