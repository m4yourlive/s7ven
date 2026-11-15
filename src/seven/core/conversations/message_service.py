
"""Conversation message persistence service."""

from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.database.models import Message
from seven.core.database.repositories import (
    ConversationRepository,
    MessageRepository,
)


class MessageService:
    """Manage persistent messages within conversations."""

    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the message service."""
        self._conversation_repository = ConversationRepository(session)
        self._message_repository = MessageRepository(session)

    async def create(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Message:
        """Persist a message in an existing conversation."""
        if role not in self.VALID_ROLES:
            raise ValueError(f"Unsupported message role: {role}")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content cannot be empty.")

        conversation = await self._conversation_repository.get_by_id(
            conversation_id,
        )

        if conversation is None:
            raise ValueError(
                f"Conversation '{conversation_id}' does not exist."
            )

        return await self._message_repository.create(
            conversation_id=conversation_id,
            role=role,
            content=content.strip(),
        )

    async def get_messages(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Retrieve messages belonging to a conversation."""
        conversation = await self._conversation_repository.get_by_id(
            conversation_id,
        )

        if conversation is None:
            raise ValueError(
                f"Conversation '{conversation_id}' does not exist."
            )

        return await self._message_repository.get_by_conversation(
            conversation_id,
            limit=limit,
            offset=offset,
        )

    async def search_messages(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        role: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Message]:
        """Search messages by content with optional filters."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if role is not None and role not in self.VALID_ROLES:
            raise ValueError(f"Unsupported message role: {role}")

        if conversation_id is not None:
            conversation = await self._conversation_repository.get_by_id(
                conversation_id,
            )

            if conversation is None:
                raise ValueError(
                    f"Conversation '{conversation_id}' does not exist."
                )

        return await self._message_repository.search(
            query.strip(),
            conversation_id=conversation_id,
            role=role,
            limit=limit,
            offset=offset,
        )

    async def delete_messages(self, conversation_id: str) -> int:
        """Delete all messages in a conversation."""
        conversation = await self._conversation_repository.get_by_id(
            conversation_id,
        )

        if conversation is None:
            raise ValueError(
                f"Conversation '{conversation_id}' does not exist."
            )

        return await self._message_repository.delete_by_conversation(
            conversation_id,
        )
