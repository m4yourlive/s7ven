
"""End-to-end conversation and memory integration tests."""

import pytest
import pytest_asyncio

from seven.core.conversations import (
    ConversationService,
    MemoryConsolidator,
    MemoryService,
    MessageService,
)
from seven.core.database import DatabaseConfig, DatabaseManager


@pytest.fixture
def database_config(tmp_path):
    return DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="conversation_memory_test.db",
    )


@pytest_asyncio.fixture
async def database_manager(database_config):
    manager = DatabaseManager(database_config)
    await manager.initialize()
    yield manager
    await manager.dispose()


@pytest.mark.asyncio
async def test_conversation_memory_lifecycle(database_manager) -> None:
    async with database_manager.session_manager.session() as session:
        conversation_service = ConversationService(session)
        message_service = MessageService(session)
        memory_service = MemoryService(session)

        consolidator = MemoryConsolidator(
            message_service=message_service,
            memory_service=memory_service,
        )

        conversation = await conversation_service.create(
            title="S7VEN Development",
        )

        await message_service.create(
            conversation_id=conversation.id,
            role="user",
            content=(
                "My long-term goal is to become "
                "a software developer."
            ),
        )

        await message_service.create(
            conversation_id=conversation.id,
            role="assistant",
            content="That is a valuable long-term goal.",
        )

        await session.commit()

        history = await message_service.get_messages(
            conversation.id,
        )

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[0].created_at < history[1].created_at

        consolidated = await consolidator.consolidate(
            conversation.id,
        )

        assert len(consolidated) == 1

        await session.commit()

        memories = await memory_service.search(
            "My long-term goal is to become a software developer.",
            limit=20,
        )

        assert len(memories) == 1
        assert memories[0].category == "goal"
        assert memories[0].conversation_id == conversation.id

        await consolidator.consolidate(conversation.id)
        await session.commit()

        memories_after_second_consolidation = (
            await memory_service.search(
                "My long-term goal is to become a software developer.",
                limit=20,
            )
        )

        assert len(memories_after_second_consolidation) == 1
