
"""Conversation management service."""

from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.database.models import Conversation
from seven.core.database.repositories import ConversationRepository


class ConversationService:
    """Manage conversation lifecycle and operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the conversation service."""
        self._repository = ConversationRepository(session)

    async def create(
        self,
        *,
        title: str = "New Conversation",
        description: str | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        if not isinstance(title, str):
            raise TypeError("Conversation title must be a string.")

        normalized_title = title.strip()

        if not normalized_title:
            raise ValueError("Conversation title cannot be empty.")

        if len(normalized_title) > 255:
            raise ValueError(
                "Conversation title cannot exceed 255 characters."
            )

        return await self._repository.create(
            title=normalized_title,
            description=description,
        )

    async def get(self, conversation_id: str) -> Conversation | None:
        """Retrieve a conversation by ID."""
        return await self._repository.get_by_id(conversation_id)

    async def get_recent(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        """Retrieve recent conversations."""
        return await self._repository.get_recent(
            limit=limit,
            offset=offset,
        )

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

        return await self._repository.search(
            query.strip(),
            limit=limit,
            offset=offset,
        )

    async def rename(
        self,
        conversation_id: str,
        new_title: str,
    ) -> Conversation | None:
        """Rename an existing conversation."""
        if not isinstance(new_title, str):
            raise TypeError("Conversation title must be a string.")

        normalized_title = new_title.strip()

        if not normalized_title:
            raise ValueError("Conversation title cannot be empty.")

        if len(normalized_title) > 255:
            raise ValueError(
                "Conversation title cannot exceed 255 characters."
            )

        conversation = await self._repository.get_by_id(conversation_id)

        if conversation is None:
            return None

        return await self._repository.update(
            conversation,
            {"title": normalized_title},
        )

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation by ID."""
        return await self._repository.delete_by_id(conversation_id)
