
"""End-to-end tests for the S7VEN conversation pipeline."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import seven.gui.bridge as bridge_module
from seven.core.ai.engine import AIEngine
from seven.core.ai.models import GenerationResponse, MessageRole
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.message_service import MessageService
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)
from seven.core.database.repositories.user_profile_repository import (
    UserProfileRepository,
)


class FakeNestedTransaction:
    """Simulate a nested database transaction."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class FakeSession:
    """Provide the asynchronous session operations required by the bridge."""

    def __init__(self):
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.nested_transaction_count = 0

    def begin_nested(self):
        self.nested_transaction_count += 1
        return FakeNestedTransaction()


class FakeSessionContext:
    """Provide an asynchronous session context manager."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class FakeSessionManager:
    """Provide a fake database session manager."""

    def __init__(self, session):
        self._session = session

    def session(self):
        return FakeSessionContext(self._session)


class FakeMemoryConsolidator:
    """Provide a deterministic memory consolidation result."""

    def __init__(self, message_service, memory_service):
        self.message_service = message_service
        self.memory_service = memory_service

    async def consolidate_detailed(self, conversation_id):
        return SimpleNamespace(
            created=0,
            reinforced=0,
            updated=0,
            kept=0,
            needs_review=0,
        )


@pytest.fixture
def end_to_end_components(monkeypatch):
    """Build the real profile service and AI pipeline."""

    session = FakeSession()
    session_manager = FakeSessionManager(session)

    conversation_service = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(id="conversation-1")
        ),
        create=AsyncMock(),
    )

    message_service = AsyncMock(spec=MessageService)

    profile_repository = AsyncMock(spec=UserProfileRepository)

    persisted_profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="prefers concise responses",
                confidence=0.95,
            ),
        ),
    )

    # Match the real repository contract:
    # save_profile() returns the persisted profile snapshot.
    profile_repository.save_profile = AsyncMock(
        return_value=persisted_profile,
    )
    profile_repository.get_profile = AsyncMock(
        return_value=persisted_profile,
    )

    memory_service = MemoryService(session=AsyncMock())

    memory_service._repository.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="memory-1",
                content=(
                    "S7VEN is a personal AI assistant "
                    "built with Python."
                ),
                category="project",
                importance=1.0,
                created_at=datetime.now(timezone.utc),
            )
        ]
    )

    config = AIConfig()

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=PromptManager(DEFAULT_PROMPTS),
    )

    provider = OpenAIProvider(api_key="test-api-key")

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="S7VEN is a personal AI assistant built with Python.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    pipeline = AIResponsePipeline(
        engine=AIEngine(provider),
        request_factory=request_factory,
    )

    monkeypatch.setattr(
        bridge_module,
        "ConversationService",
        lambda database_session: conversation_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "MessageService",
        lambda database_session: message_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "MemoryService",
        lambda database_session: memory_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "UserProfileRepository",
        lambda database_session: profile_repository,
    )
    monkeypatch.setattr(
        bridge_module,
        "MemoryConsolidator",
        FakeMemoryConsolidator,
    )

    bridge = bridge_module.FrontendBridge(
        ai_pipeline_provider=lambda: pipeline,
        database_session_manager_provider=lambda: session_manager,
    )

    return {
        "bridge": bridge,
        "pipeline": pipeline,
        "provider": provider,
        "session": session,
        "profile_repository": profile_repository,
        "persisted_profile": persisted_profile,
        "memory_service": memory_service,
        "message_service": message_service,
    }


@pytest.mark.asyncio
async def test_end_to_end_extracts_profile_and_personalizes_response(
    end_to_end_components,
) -> None:
    components = end_to_end_components

    result = await components["bridge"]._send_message_async(
        message=(
            "I prefer concise responses. "
            "Explain S7VEN's Python architecture."
        ),
        conversation_id="conversation-1",
    )

    assert result["success"] is True
    assert result["reply"] == (
        "S7VEN is a personal AI assistant built with Python."
    )

    repository = components["profile_repository"]

    repository.save_profile.assert_awaited_once()

    extracted_profile = repository.save_profile.await_args.args[0]

    assert isinstance(extracted_profile, UserProfile)
    assert any(
        "concise" in str(preference.value).casefold()
        for preference in extracted_profile.preferences
    )

    request = components["provider"].generate.call_args.args[0]

    memory_messages = [
        message
        for message in request.messages
        if (
            message.role is MessageRole.SYSTEM
            and "S7VEN is a personal AI assistant" in message.content
        )
    ]

    personalization_messages = [
        message
        for message in request.messages
        if (
            message.role is MessageRole.SYSTEM
            and "Response style instructions:" in message.content
        )
    ]

    assert len(memory_messages) == 1
    assert len(personalization_messages) == 1

    personalization_context = personalization_messages[0].content

    assert "Keep communication concise" in personalization_context

    assert request.messages[-1].role is MessageRole.USER
    assert request.messages[-1].content == (
        "I prefer concise responses. "
        "Explain S7VEN's Python architecture."
    )

    components["memory_service"]._repository.search.assert_awaited_once()

    assert components["pipeline"].context.message_count == 2
    assert components["session"].nested_transaction_count == 2

    components["session"].commit.assert_awaited_once()
    components["session"].rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_end_to_end_reuses_existing_profile(
    end_to_end_components,
) -> None:
    components = end_to_end_components
    repository = components["profile_repository"]

    result = await components["bridge"]._send_message_async(
        message="Explain S7VEN's Python architecture.",
        conversation_id="conversation-1",
    )

    assert result["success"] is True

    repository.get_profile.assert_awaited_once()
    repository.save_profile.assert_not_awaited()

    request = components["provider"].generate.call_args.args[0]

    personalization_messages = [
        message
        for message in request.messages
        if (
            message.role is MessageRole.SYSTEM
            and "Response style instructions:" in message.content
        )
    ]

    assert len(personalization_messages) == 1
    assert "Keep communication concise" in (
        personalization_messages[0].content
    )

    assert components["pipeline"].context.message_count == 2
    components["session"].commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_end_to_end_keeps_ai_available_without_saved_profile(
    end_to_end_components,
) -> None:
    components = end_to_end_components
    repository = components["profile_repository"]

    repository.get_profile.return_value = UserProfile()

    result = await components["bridge"]._send_message_async(
        message="What is S7VEN?",
        conversation_id="conversation-1",
    )

    assert result["success"] is True

    request = components["provider"].generate.call_args.args[0]

    personalization_messages = [
        message
        for message in request.messages
        if (
            message.role is MessageRole.SYSTEM
            and "Response style instructions:" in message.content
        )
    ]

    assert personalization_messages == []
    assert components["pipeline"].context.message_count == 2
    components["session"].commit.assert_awaited_once()
