
"""Long-term memory management service."""

from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.database.models import Memory
from seven.core.database.repositories import MemoryRepository


class MemoryService:
    """Manage persistent long-term memories."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the memory service."""
        self._repository = MemoryRepository(session)

    async def create(
        self,
        *,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        conversation_id: str | None = None,
    ) -> Memory:
        """Create a persistent memory."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Memory content cannot be empty.")

        if not isinstance(category, str) or not category.strip():
            raise ValueError("Memory category cannot be empty.")

        if (
            not isinstance(importance, (int, float))
            or isinstance(importance, bool)
            or not 0.0 <= importance <= 1.0
        ):
            raise ValueError(
                "Memory importance must be between 0.0 and 1.0."
            )

        return await self._repository.create(
            content=content.strip(),
            category=category.strip().lower(),
            importance=float(importance),
            conversation_id=conversation_id,
        )

    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        return await self._repository.get_by_id(memory_id)

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[Memory]:
        """Search persistent memories by content."""
        return await self._repository.search(
            query,
            limit=limit,
        )

    async def get_by_category(
        self,
        category: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Retrieve memories by category."""
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Memory category cannot be empty.")

        return await self._repository.get_by_category(
            category.strip().lower(),
            limit=limit,
            offset=offset,
        )

    async def get_important(
        self,
        *,
        min_importance: float = 0.7,
        limit: int = 20,
    ) -> list[Memory]:
        """Retrieve memories above an importance threshold."""
        return await self._repository.get_important(
            min_importance=min_importance,
            limit=limit,
        )

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: str | None = None,
        importance: float | None = None,
    ) -> Memory | None:
        """Update an existing memory."""
        memory = await self._repository.get_by_id(memory_id)

        if memory is None:
            return None

        values: dict[str, str | float] = {}

        if content is not None:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Memory content cannot be empty.")

            values["content"] = content.strip()

        if category is not None:
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Memory category cannot be empty.")

            values["category"] = category.strip().lower()

        if importance is not None:
            if (
                not isinstance(importance, (int, float))
                or isinstance(importance, bool)
                or not 0.0 <= importance <= 1.0
            ):
                raise ValueError(
                    "Memory importance must be between 0.0 and 1.0."
                )

            values["importance"] = float(importance)

        if not values:
            return memory

        return await self._repository.update(memory, values)

    async def delete(self, memory_id: str) -> bool:
        """Delete a persistent memory."""
        return await self._repository.delete_by_id(memory_id)
