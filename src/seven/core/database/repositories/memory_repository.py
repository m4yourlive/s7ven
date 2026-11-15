"""Memory repository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.database.models import Memory
from seven.core.database.repositories.base_repository import BaseRepository


class MemoryRepository(BaseRepository[Memory]):
    """Provide database operations for long-term memories."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the memory repository."""
        super().__init__(session, Memory)

    async def create(
        self,
        *,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        conversation_id: str | None = None,
    ) -> Memory:
        """Create a new memory."""
        if not 0.0 <= importance <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0.")

        memory = Memory(
            content=content,
            category=category,
            importance=importance,
            conversation_id=conversation_id,
        )

        return await self.add(memory)

    async def get_by_category(
        self,
        category: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Retrieve memories belonging to a category."""
        if limit is not None and limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        statement = (
            select(Memory)
            .where(Memory.category == category)
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .offset(offset)
        )

        if limit is not None:
            statement = statement.limit(limit)

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def get_important(
        self,
        *,
        min_importance: float = 0.7,
        limit: int = 20,
    ) -> list[Memory]:
        """Retrieve memories above a minimum importance threshold."""
        if not 0.0 <= min_importance <= 1.0:
            raise ValueError("Minimum importance must be between 0.0 and 1.0.")

        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        statement = (
            select(Memory)
            .where(Memory.importance >= min_importance)
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[Memory]:
        """Search memories by content using a case-insensitive match."""
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        statement = (
            select(Memory)
            .where(Memory.content.ilike(f"%{query.strip()}%"))
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())
