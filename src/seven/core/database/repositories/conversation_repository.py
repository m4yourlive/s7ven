
"""Conversation repository."""

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from seven.core.database.models import Conversation
from seven.core.database.repositories.base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Provide database operations for conversations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the conversation repository."""
        super().__init__(session, Conversation)

    async def create(
        self,
        *,
        title: str = "New Conversation",
        description: str | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            title=title,
            description=description,
        )

        return await self.add(conversation)

    async def get_by_id(
        self,
        entity_id: str,
    ) -> Conversation | None:
        """Retrieve a conversation with its messages."""
        statement = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == entity_id)
        )

        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def get_recent(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        """Retrieve recent conversations."""
        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        statement = (
            select(Conversation)
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.created_at.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        """Search conversations by title or description."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        pattern = f"%{query.strip()}%"

        statement = (
            select(Conversation)
            .where(
                or_(
                    Conversation.title.ilike(pattern),
                    Conversation.description.ilike(pattern),
                )
            )
            .order_by(
                Conversation.updated_at.desc(),
                Conversation.created_at.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())
