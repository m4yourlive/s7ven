
"""Tests for automatic profile integration in the frontend bridge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import seven.gui.bridge as bridge_module
from seven.gui.bridge import FrontendBridge
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


class FakeNestedTransaction:
    """Simulate a nested database transaction."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class FakeSession:
    """Provide the session operations required by the bridge."""

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


def build_bridge(
    monkeypatch,
    *,
    profile_service,
):
    """Build a bridge with mocked database and AI dependencies."""
    session = FakeSession()
    session_manager = FakeSessionManager(session)

    conversation_service = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(id="conversation-1")
        ),
        create=AsyncMock(),
    )

    message_service = object()
    memory_service = object()
    memory_retriever = object()
    context_injector = object()
    profile_repository = object()

    pipeline = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                content="Personalized response.",
                model="test-model",
                provider="test-provider",
                finish_reason="stop",
                usage=None,
            )
        )
    )

    monkeypatch.setattr(
        bridge_module,
        "ConversationService",
        lambda session: conversation_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "MessageService",
        lambda session: message_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "MemoryService",
        lambda session: memory_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "MemoryRetriever",
        lambda service: memory_retriever,
    )
    monkeypatch.setattr(
        bridge_module,
        "ContextInjector",
        lambda retriever: context_injector,
    )
    monkeypatch.setattr(
        bridge_module,
        "UserProfileRepository",
        lambda session: profile_repository,
    )
    monkeypatch.setattr(
        bridge_module,
        "UserProfileService",
        lambda repository: profile_service,
    )
    monkeypatch.setattr(
        bridge_module,
        "MemoryConsolidator",
        FakeMemoryConsolidator,
    )

    bridge = FrontendBridge(
        ai_pipeline_provider=lambda: pipeline,
        database_session_manager_provider=lambda: session_manager,
    )

    return {
        "bridge": bridge,
        "pipeline": pipeline,
        "session": session,
        "profile_service": profile_service,
        "profile_repository": profile_repository,
        "conversation_service": conversation_service,
        "message_service": message_service,
        "context_injector": context_injector,
    }


@pytest.mark.asyncio
async def test_bridge_extracts_and_passes_user_profile(
    monkeypatch,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="prefers concise responses",
                confidence=0.95,
            ),
        ),
    )

    profile_service = SimpleNamespace(
        extract_and_save=AsyncMock(return_value=profile)
    )

    components = build_bridge(
        monkeypatch,
        profile_service=profile_service,
    )

    result = await components["bridge"]._send_message_async(
        message="I prefer concise responses.",
        conversation_id="conversation-1",
    )

    assert result["success"] is True
    assert result["reply"] == "Personalized response."

    profile_service.extract_and_save.assert_awaited_once_with(
        "I prefer concise responses."
    )

    pipeline_call = components["pipeline"].generate.await_args

    assert pipeline_call.args[0] == "I prefer concise responses."
    assert pipeline_call.kwargs["user_profile"] is profile
    assert pipeline_call.kwargs["message_service"] is (
        components["message_service"]
    )
    assert pipeline_call.kwargs["context_injector"] is (
        components["context_injector"]
    )

    assert components["session"].nested_transaction_count == 2
    components["session"].commit.assert_awaited_once()
    components["session"].rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_continues_without_profile_when_extraction_fails(
    monkeypatch,
) -> None:
    profile_service = SimpleNamespace(
        extract_and_save=AsyncMock(
            side_effect=RuntimeError("Profile storage unavailable.")
        )
    )

    components = build_bridge(
        monkeypatch,
        profile_service=profile_service,
    )

    result = await components["bridge"]._send_message_async(
        message="Explain Python.",
        conversation_id="conversation-1",
    )

    assert result["success"] is True

    pipeline_call = components["pipeline"].generate.await_args

    assert pipeline_call.kwargs["user_profile"] is None

    components["pipeline"].generate.assert_awaited_once()
    components["session"].commit.assert_awaited_once()
    components["session"].rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_uses_profile_for_new_conversations(
    monkeypatch,
) -> None:
    profile = UserProfile(
        preferences=(
            UserPreference(
                key="response_style",
                value="concise",
            ),
        ),
    )

    profile_service = SimpleNamespace(
        extract_and_save=AsyncMock(return_value=profile)
    )

    components = build_bridge(
        monkeypatch,
        profile_service=profile_service,
    )

    components["conversation_service"].get.return_value = None
    components["conversation_service"].create.return_value = (
        SimpleNamespace(id="new-conversation")
    )

    result = await components["bridge"]._send_message_async(
        message="Hello S7VEN.",
    )

    assert result["success"] is True
    assert result["conversation_id"] == "new-conversation"

    pipeline_call = components["pipeline"].generate.await_args

    assert pipeline_call.kwargs["user_profile"] is profile
    components["session"].commit.assert_awaited_once()
