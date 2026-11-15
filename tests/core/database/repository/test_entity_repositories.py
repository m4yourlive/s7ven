"""Tests for conversation, message, and memory repositories."""

import pytest
import pytest_asyncio

from seven.core.database import Base, DatabaseConfig, DatabaseSessionManager
from seven.core.database.models import Conversation, Memory, Message
from seven.core.database.repositories import (
    ConversationRepository,
    MemoryRepository,
    MessageRepository,
)


@pytest.fixture
def database_config(tmp_path):
    """Provide an isolated database configuration."""
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="entity_repository_test.db",
    )


@pytest_asyncio.fixture
async def repositories(database_config):
    """Provide initialized repositories and a database session."""
    manager = DatabaseSessionManager(database_config)

    await manager.initialize()

    async with manager.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with manager.session() as session:
        yield (
            ConversationRepository(session),
            MessageRepository(session),
            MemoryRepository(session),
        )

    await manager.dispose()


class TestConversationRepository:
    @pytest.mark.asyncio
    async def test_create_conversation(self, repositories):
        conversation_repo, _, _ = repositories

        conversation = await conversation_repo.create(
            title="S7VEN Development",
            description="Project discussion",
        )

        assert conversation.id is not None
        assert conversation.title == "S7VEN Development"
        assert conversation.description == "Project discussion"

    @pytest.mark.asyncio
    async def test_create_conversation_with_defaults(self, repositories):
        conversation_repo, _, _ = repositories

        conversation = await conversation_repo.create()

        assert conversation.title == "New Conversation"
        assert conversation.description is None

    @pytest.mark.asyncio
    async def test_get_by_id_loads_messages(self, repositories):
        conversation_repo, message_repo, _ = repositories

        conversation = await conversation_repo.create(
            title="Test Conversation",
        )

        await message_repo.create(
            conversation_id=conversation.id,
            role="user",
            content="Hello!",
        )

        result = await conversation_repo.get_by_id(conversation.id)

        assert result is not None
        assert len(result.messages) == 1
        assert result.messages[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_missing_conversation(
        self,
        repositories,
    ):
        conversation_repo, _, _ = repositories

        result = await conversation_repo.get_by_id("missing-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_supports_limit(self, repositories):
        conversation_repo, _, _ = repositories

        await conversation_repo.create(title="First")
        await conversation_repo.create(title="Second")

        result = await conversation_repo.get_recent(limit=1)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_recent_rejects_invalid_limit(self, repositories):
        conversation_repo, _, _ = repositories

        with pytest.raises(ValueError, match="Limit"):
            await conversation_repo.get_recent(limit=0)

    @pytest.mark.asyncio
    async def test_get_recent_rejects_negative_offset(self, repositories):
        conversation_repo, _, _ = repositories

        with pytest.raises(ValueError, match="Offset"):
            await conversation_repo.get_recent(offset=-1)


class TestMessageRepository:
    @pytest.mark.asyncio
    async def test_create_message(self, repositories):
        conversation_repo, message_repo, _ = repositories

        conversation = await conversation_repo.create()

        message = await message_repo.create(
            conversation_id=conversation.id,
            role="user",
            content="Hello, S7VEN!",
        )

        assert message.id is not None
        assert message.conversation_id == conversation.id
        assert message.role == "user"
        assert message.content == "Hello, S7VEN!"

    @pytest.mark.asyncio
    async def test_get_by_conversation(self, repositories):
        conversation_repo, message_repo, _ = repositories

        first = await conversation_repo.create(title="First")
        second = await conversation_repo.create(title="Second")

        await message_repo.create(
            conversation_id=first.id,
            role="user",
            content="First message",
        )
        await message_repo.create(
            conversation_id=first.id,
            role="assistant",
            content="Second message",
        )
        await message_repo.create(
            conversation_id=second.id,
            role="user",
            content="Other conversation",
        )

        result = await message_repo.get_by_conversation(first.id)

        assert len(result) == 2
        assert {message.content for message in result} == {
            "First message",
            "Second message",
        }

    @pytest.mark.asyncio
    async def test_get_by_conversation_supports_pagination(
        self,
        repositories,
    ):
        conversation_repo, message_repo, _ = repositories

        conversation = await conversation_repo.create()

        for index in range(3):
            await message_repo.create(
                conversation_id=conversation.id,
                role="user",
                content=f"Message {index}",
            )

        result = await message_repo.get_by_conversation(
            conversation.id,
            limit=1,
            offset=1,
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_by_conversation_rejects_invalid_limit(
        self,
        repositories,
    ):
        _, message_repo, _ = repositories

        with pytest.raises(ValueError, match="Limit"):
            await message_repo.get_by_conversation(
                "conversation-id",
                limit=0,
            )

    @pytest.mark.asyncio
    async def test_get_by_conversation_rejects_negative_offset(
        self,
        repositories,
    ):
        _, message_repo, _ = repositories

        with pytest.raises(ValueError, match="Offset"):
            await message_repo.get_by_conversation(
                "conversation-id",
                offset=-1,
            )

    @pytest.mark.asyncio
    async def test_delete_by_conversation(self, repositories):
        conversation_repo, message_repo, _ = repositories

        conversation = await conversation_repo.create()

        for index in range(3):
            await message_repo.create(
                conversation_id=conversation.id,
                role="user",
                content=f"Message {index}",
            )

        deleted = await message_repo.delete_by_conversation(
            conversation.id,
        )

        remaining = await message_repo.get_by_conversation(
            conversation.id,
        )

        assert deleted == 3
        assert remaining == []


class TestMemoryRepository:
    @pytest.mark.asyncio
    async def test_create_memory(self, repositories):
        _, _, memory_repo = repositories

        memory = await memory_repo.create(
            content="User is building S7VEN.",
            category="project",
            importance=0.9,
        )

        assert memory.id is not None
        assert memory.content == "User is building S7VEN."
        assert memory.category == "project"
        assert memory.importance == 0.9

    @pytest.mark.asyncio
    async def test_create_memory_with_defaults(self, repositories):
        _, _, memory_repo = repositories

        memory = await memory_repo.create(
            content="General memory",
        )

        assert memory.category == "general"
        assert memory.importance == 0.5
        assert memory.conversation_id is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("importance", [-0.1, 1.1])
    async def test_create_memory_rejects_invalid_importance(
        self,
        repositories,
        importance,
    ):
        _, _, memory_repo = repositories

        with pytest.raises(ValueError, match="Importance"):
            await memory_repo.create(
                content="Invalid memory",
                importance=importance,
            )

    @pytest.mark.asyncio
    async def test_get_by_category(self, repositories):
        _, _, memory_repo = repositories

        await memory_repo.create(
            content="Project memory",
            category="project",
            importance=0.9,
        )
        await memory_repo.create(
            content="General memory",
            category="general",
        )

        result = await memory_repo.get_by_category("project")

        assert len(result) == 1
        assert result[0].content == "Project memory"

    @pytest.mark.asyncio
    async def test_get_by_category_supports_pagination(self, repositories):
        _, _, memory_repo = repositories

        for index in range(3):
            await memory_repo.create(
                content=f"Memory {index}",
                category="project",
            )

        result = await memory_repo.get_by_category(
            "project",
            limit=1,
            offset=1,
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_important(self, repositories):
        _, _, memory_repo = repositories

        await memory_repo.create(
            content="Important memory",
            importance=0.9,
        )
        await memory_repo.create(
            content="Less important memory",
            importance=0.3,
        )

        result = await memory_repo.get_important(
            min_importance=0.7,
        )

        assert len(result) == 1
        assert result[0].content == "Important memory"

    @pytest.mark.asyncio
    async def test_get_important_rejects_invalid_threshold(
        self,
        repositories,
    ):
        _, _, memory_repo = repositories

        with pytest.raises(ValueError, match="importance"):
            await memory_repo.get_important(min_importance=1.5)

    @pytest.mark.asyncio
    async def test_get_important_rejects_invalid_limit(self, repositories):
        _, _, memory_repo = repositories

        with pytest.raises(ValueError, match="Limit"):
            await memory_repo.get_important(limit=0)

    @pytest.mark.asyncio
    async def test_search_memory(self, repositories):
        _, _, memory_repo = repositories

        await memory_repo.create(
            content="S7VEN is an AI companion.",
            importance=0.9,
        )
        await memory_repo.create(
            content="Learning Python programming.",
            importance=0.5,
        )

        result = await memory_repo.search("s7ven")

        assert len(result) == 1
        assert "S7VEN" in result[0].content

    @pytest.mark.asyncio
    async def test_search_rejects_empty_query(self, repositories):
        _, _, memory_repo = repositories

        with pytest.raises(ValueError, match="Search query"):
            await memory_repo.search("   ")

    @pytest.mark.asyncio
    async def test_search_rejects_invalid_limit(self, repositories):
        _, _, memory_repo = repositories

        with pytest.raises(ValueError, match="Limit"):
            await memory_repo.search("memory", limit=0)
