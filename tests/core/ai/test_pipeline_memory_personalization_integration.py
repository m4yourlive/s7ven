
"""End-to-end tests for memory and personalization integration."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.models import (
    GenerationResponse,
    MessageRole,
)
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig
from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


@pytest.fixture
def pipeline_components():
    """Build a pipeline with a mockable AI provider."""
    config = AIConfig()
    prompt_manager = PromptManager(DEFAULT_PROMPTS)

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=prompt_manager,
    )

    provider = OpenAIProvider(api_key="test-api-key")
    engine = AIEngine(provider)

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
    )

    return pipeline, provider


@pytest.fixture
def memory_context_injector():
    """Build a context injector with a mocked memory repository."""
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
                created_at=datetime.now(UTC),
            )
        ]
    )

    retriever = MemoryRetriever(memory_service)

    injector = ContextInjector(
        retriever,
        max_memories=5,
    )

    return injector, memory_service._repository.search


@pytest.fixture
def user_profile() -> UserProfile:
    """Build a profile containing multiple personalization preferences."""
    return UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="prefers concise responses",
                confidence=0.95,
            ),
            UserPreference(
                key="technical_depth",
                value="beginner",
                confidence=0.9,
            ),
            UserPreference(
                key="learning_style",
                value="practical",
                confidence=0.9,
            ),
            UserPreference(
                key="formatting",
                value="step_by_step",
                confidence=0.9,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_generate_combines_memory_and_personalization(
    pipeline_components,
    memory_context_injector,
    user_profile,
) -> None:
    pipeline, provider = pipeline_components
    injector, search_mock = memory_context_injector

    expected_response = GenerationResponse(
        content="S7VEN is a Python-based personal AI assistant.",
        model="gpt-4o-mini",
        provider="openai",
    )

    provider.generate = AsyncMock(return_value=expected_response)

    response = await pipeline.generate(
        "Explain S7VEN's Python architecture step by step.",
        context_injector=injector,
        user_profile=user_profile,
    )

    assert response == expected_response
    search_mock.assert_awaited_once()

    request = provider.generate.call_args.args[0]

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
    assert "fundamentals" in personalization_context
    assert "practical explanations" in personalization_context
    assert "sequential steps" in personalization_context

    assert request.messages[-1].role is MessageRole.USER
    assert request.messages[-1].content == (
        "Explain S7VEN's Python architecture step by step."
    )


@pytest.mark.asyncio
async def test_stream_combines_memory_and_personalization(
    pipeline_components,
    memory_context_injector,
    user_profile,
) -> None:
    pipeline, provider = pipeline_components
    injector, search_mock = memory_context_injector

    captured_request = None

    async def mock_stream(request):
        nonlocal captured_request
        captured_request = request

        yield "S7VEN is a "
        yield "Python-based AI assistant."

    provider.stream = mock_stream

    chunks = [
        chunk
        async for chunk in pipeline.stream(
            "Explain S7VEN's Python architecture step by step.",
            context_injector=injector,
            user_profile=user_profile,
        )
    ]

    assert chunks == [
        "S7VEN is a ",
        "Python-based AI assistant.",
    ]

    search_mock.assert_awaited_once()
    assert captured_request is not None

    memory_messages = [
        message
        for message in captured_request.messages
        if (
            message.role is MessageRole.SYSTEM
            and "S7VEN is a personal AI assistant" in message.content
        )
    ]

    personalization_messages = [
        message
        for message in captured_request.messages
        if (
            message.role is MessageRole.SYSTEM
            and "Response style instructions:" in message.content
        )
    ]

    assert len(memory_messages) == 1
    assert len(personalization_messages) == 1

    personalization_context = personalization_messages[0].content

    assert "Keep communication concise" in personalization_context
    assert "fundamentals" in personalization_context
    assert "practical explanations" in personalization_context
    assert "sequential steps" in personalization_context

    assert pipeline.context.message_count == 2


@pytest.mark.asyncio
async def test_personalization_is_not_added_to_conversation_history(
    pipeline_components,
    user_profile,
) -> None:
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="A Python explanation.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    await pipeline.generate(
        "Explain Python architecture step by step.",
        user_profile=user_profile,
    )

    history = pipeline.context.history

    assert len(history) == 2
    assert history[0].role is MessageRole.USER
    assert history[1].role is MessageRole.ASSISTANT

    assert all(
        "Personalization instructions:" not in message.content
        for message in history
    )


@pytest.mark.asyncio
async def test_memory_injection_works_without_user_profile(
    pipeline_components,
    memory_context_injector,
) -> None:
    pipeline, provider = pipeline_components
    injector, search_mock = memory_context_injector

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="S7VEN is a personal AI assistant.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    await pipeline.generate(
        "Tell me about S7VEN.",
        context_injector=injector,
    )

    request = provider.generate.call_args.args[0]

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
    assert personalization_messages == []

    search_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_personalization_works_without_memory_injector(
    pipeline_components,
    user_profile,
) -> None:
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock(
        return_value=GenerationResponse(
            content="Here is a Python explanation.",
            model="gpt-4o-mini",
            provider="openai",
        )
    )

    await pipeline.generate(
        "Explain Python architecture step by step.",
        user_profile=user_profile,
    )

    request = provider.generate.call_args.args[0]

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
