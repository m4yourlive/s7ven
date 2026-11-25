"""Integration tests for automatic memory consolidation in the bridge."""

from types import SimpleNamespace

import pytest
import pytest_asyncio

from seven.core.conversations import MemoryService
from seven.core.database import DatabaseConfig, DatabaseManager
from seven.gui.bridge import FrontendBridge


@pytest_asyncio.fixture
async def database_manager(tmp_path):
    """Create an isolated database for bridge integration tests."""
    config = DatabaseConfig(
        data_directory=tmp_path / "database",
        database_name="bridge_memory_test.db",
    )

    manager = DatabaseManager(config)
    await manager.initialize()

    yield manager

    await manager.dispose()


class FakeAIPipeline:
    """Persist a simulated exchange and return an AI response."""

    async def generate(
        self,
        message,
        *,
        message_service=None,
        conversation_id=None,
        context_injector=None,
        user_profile=None,
    ):
        await message_service.create(
            conversation_id=conversation_id,
            role="user",
            content=message,
        )

        await message_service.create(
            conversation_id=conversation_id,
            role="assistant",
            content="I understand your long-term goal.",
        )

        return SimpleNamespace(
            content="I understand your long-term goal.",
            model="test-model",
            provider="test-provider",
            finish_reason="stop",
            usage={"total_tokens": 20},
        )


@pytest.mark.asyncio
async def test_bridge_automatically_consolidates_user_memory(
    database_manager,
):
    """Verify that a successful exchange creates persistent memory."""
    pipeline = FakeAIPipeline()

    bridge = FrontendBridge(
        ai_pipeline_provider=lambda: pipeline,
        database_session_manager_provider=(
            lambda: database_manager.session_manager
        ),
    )

    message = (
        "My long-term goal is to become a software developer."
    )

    result = await bridge._send_message_async(message)

    assert result["success"] is True
    assert result["reply"] == (
        "I understand your long-term goal."
    )
    assert result["conversation_id"]

    async with database_manager.session_manager.session() as session:
        memory_service = MemoryService(session)

        memories = await memory_service.search(
            message,
            limit=20,
        )

        assert len(memories) == 1
        assert memories[0].category == "goal"
        assert memories[0].conversation_id == (
            result["conversation_id"]
        )
        assert memories[0].content == message


@pytest.mark.asyncio
async def test_bridge_does_not_duplicate_existing_memory(
    database_manager,
):
    """Verify repeated consolidation does not create duplicate memories."""
    pipeline = FakeAIPipeline()

    bridge = FrontendBridge(
        ai_pipeline_provider=lambda: pipeline,
        database_session_manager_provider=(
            lambda: database_manager.session_manager
        ),
    )

    message = (
        "My long-term goal is to become a software developer."
    )

    first_result = await bridge._send_message_async(message)

    assert first_result["success"] is True

    second_result = await bridge._send_message_async(
        message,
        conversation_id=first_result["conversation_id"],
    )

    assert second_result["success"] is True

    async with database_manager.session_manager.session() as session:
        memory_service = MemoryService(session)

        memories = await memory_service.search(
            message,
            limit=20,
        )

        assert len(memories) == 1