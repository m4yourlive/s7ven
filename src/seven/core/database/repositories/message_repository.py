
"""Message repository."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.database.models import Message
from seven.core.database.repositories.base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Provide database operations for conversation messages."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the message repository."""
        super().__init__(session, Message)

    async def create(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Message:
        """Create a new conversation message."""
        previous_message = await self.session.scalar(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(1)
        )

        timestamp = datetime.now(UTC)

        if previous_message is not None:
            previous_timestamp = previous_message.created_at

            if previous_timestamp.tzinfo is None:
                previous_timestamp = previous_timestamp.replace(
                    tzinfo=UTC,
                )

            minimum_next_timestamp = (
                previous_timestamp.replace(microsecond=0)
                + timedelta(seconds=1)
            )

            timestamp = max(timestamp, minimum_next_timestamp)

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=timestamp,
        )

        return await self.add(message)

    async def get_by_conversation(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Retrieve messages belonging to a conversation."""
        if limit is not None and limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(
                Message.created_at.asc(),
                Message.id.asc(),
            )
            .offset(offset)
        )

        if limit is not None:
            statement = statement.limit(limit)

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        role: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Message]:
        """Search message content with optional conversation and role filters."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit < 1:
            raise ValueError("Limit must be greater than zero.")

        if offset < 0:
            raise ValueError("Offset cannot be negative.")

        if role is not None and role not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError(f"Unsupported message role: {role}")

        statement = select(Message).where(
            Message.content.ilike(f"%{query.strip()}%"),
        )

        if conversation_id is not None:
            statement = statement.where(
                Message.conversation_id == conversation_id,
            )

        if role is not None:
            statement = statement.where(Message.role == role)

        statement = (
            statement
            .order_by(
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())

    async def delete_by_conversation(
        self,
        conversation_id: str,
    ) -> int:
        """Delete all messages belonging to a conversation."""
        messages = await self.get_by_conversation(conversation_id)

        for message in messages:
            await self.session.delete(message)

        await self.session.flush()

        return len(messages)
