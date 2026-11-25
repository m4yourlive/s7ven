
"""Integration tests for conversation continuity across AI providers."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    MessageRole,
)
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig
from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_service import MemoryService
from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
)


class MockAIProvider(AIProvider):
    """Mock provider for testing unified AI runtime behavior."""

    def __init__(self, name: str, response_content: str) -> None:
        self._name = name
        self.response_content = response_content
        self.requests: list[GenerationRequest] = []
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def generate(
        self,
        generation_request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(generation_request)

        return GenerationResponse(
            content=self.response_content,
            model=generation_request.model,
            provider=self.name,
        )

    async def stream(
        self,
        generation_request: GenerationRequest,
    ) -> AsyncIterator[str]:
        self.requests.append(generation_request)
        yield self.response_content

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def switching_components():
    config = AIConfig(
        mode="api",
        provider="openai",
        model="gpt-4o-mini",
        local_runtime="ollama",
        local_model="llama3.2",
    )

    prompt_manager = PromptManager(DEFAULT_PROMPTS)

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=prompt_manager,
    )

    api_provider = MockAIProvider(
        name="openai",
        response_content="This response came from API AI.",
    )

    local_provider = MockAIProvider(
        name="local",
        response_content="This response came from Local AI.",
    )

    registry = ProviderRegistry()

    engine = AIEngine(
        provider=api_provider,
        provider_registry=registry,
    )

    registry.register(local_provider)

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
    )

    return pipeline, engine, api_provider, local_provider


@pytest.fixture
def memory_context_injector():
    """Build a context injector with a mocked memory repository."""
    memory_service = MemoryService(session=AsyncMock())

    memory_service._repository.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="memory-1",
                content="S7VEN is a personal AI assistant.",
                category="project",
                importance=1.0,
                created_at=datetime.now(UTC),
            )
        ]
    )

    memory_retriever = MemoryRetriever(memory_service)

    injector = ContextInjector(
        memory_retriever,
        max_memories=5,
    )

    return injector, memory_service._repository.search


@pytest.mark.asyncio
async def test_provider_switch_preserves_conversation_history(
    switching_components,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components

    first_response = await pipeline.generate(
        "Remember this conversation."
    )

    assert first_response.content == "This response came from API AI."
    assert engine.provider_name == "openai"
    assert pipeline.context.message_count == 2

    await engine.switch_provider("local")

    assert engine.provider_name == "local"

    second_response = await pipeline.generate(
        "What did you just tell me?"
    )

    assert second_response.content == "This response came from Local AI."
    assert len(api_provider.requests) == 1
    assert len(local_provider.requests) == 1

    local_messages = local_provider.requests[0].messages

    assert any(
        message.role is MessageRole.USER
        and message.content == "Remember this conversation."
        for message in local_messages
    )

    assert any(
        message.role is MessageRole.ASSISTANT
        and message.content == "This response came from API AI."
        for message in local_messages
    )

    assert local_messages[-1].role is MessageRole.USER
    assert local_messages[-1].content == "What did you just tell me?"
    assert pipeline.context.message_count == 4


@pytest.mark.asyncio
async def test_provider_switch_preserves_history_during_streaming(
    switching_components,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components

    await pipeline.generate("Initial message.")

    await engine.switch_provider("local")

    chunks = [
        chunk
        async for chunk in pipeline.stream("Continue the conversation.")
    ]

    assert chunks == ["This response came from Local AI."]
    assert len(api_provider.requests) == 1
    assert len(local_provider.requests) == 1

    local_request = local_provider.requests[0]

    assert any(
        message.role is MessageRole.ASSISTANT
        and message.content == "This response came from API AI."
        for message in local_request.messages
    )

    assert local_request.messages[-1].content == (
        "Continue the conversation."
    )
    assert pipeline.context.message_count == 4


@pytest.mark.asyncio
async def test_switching_back_to_api_preserves_local_conversation(
    switching_components,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components

    await pipeline.generate("First message.")

    await engine.switch_provider("local")
    await pipeline.generate("Second message.")

    await engine.switch_provider("openai")

    response = await pipeline.generate("Third message.")

    assert response.content == "This response came from API AI."

    api_request = api_provider.requests[-1]

    assert any(
        message.role is MessageRole.ASSISTANT
        and message.content == "This response came from Local AI."
        for message in api_request.messages
    )

    assert api_request.messages[-1].content == "Third message."
    assert pipeline.context.message_count == 6


@pytest.mark.asyncio
async def test_provider_switch_does_not_close_previous_provider(
    switching_components,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components

    await pipeline.generate("Initial message.")

    await engine.switch_provider("local")

    assert api_provider.closed is False
    assert local_provider.closed is False

    await engine.close()

    assert api_provider.closed is True
    assert local_provider.closed is True


@pytest.mark.asyncio
async def test_memory_context_survives_provider_switch(
    switching_components,
    memory_context_injector,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components
    injector, search_mock = memory_context_injector

    await engine.switch_provider("local")

    await pipeline.generate(
        "Tell me about S7VEN.",
        context_injector=injector,
    )

    assert len(local_provider.requests) == 1
    assert len(api_provider.requests) == 0

    local_messages = local_provider.requests[0].messages

    assert any(
        message.role is MessageRole.SYSTEM
        and "S7VEN is a personal AI assistant." in message.content
        for message in local_messages
    )

    search_mock.assert_awaited_once()
    assert pipeline.context.message_count == 2


@pytest.mark.asyncio
async def test_user_personalization_survives_provider_switch(
    switching_components,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components

    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="casual",
            ),
        ),
    )

    await engine.switch_provider("local")

    await pipeline.generate(
        "Explain something to me.",
        user_profile=profile,
    )

    assert len(local_provider.requests) == 1
    assert len(api_provider.requests) == 0

    local_messages = local_provider.requests[0].messages

    personalization_messages = [
        message
        for message in local_messages
        if (
            message.role is MessageRole.SYSTEM
            and "casual" in message.content.casefold()
        )
    ]

    assert personalization_messages
    assert pipeline.context.message_count == 2


@pytest.mark.asyncio
async def test_memory_and_personalization_work_together_after_switch(
    switching_components,
    memory_context_injector,
) -> None:
    pipeline, engine, api_provider, local_provider = switching_components
    injector, search_mock = memory_context_injector

    profile = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="friendly",
            ),
        ),
    )

    await engine.switch_provider("local")

    await pipeline.generate(
        "Tell me about S7VEN.",
        context_injector=injector,
        user_profile=profile,
    )

    local_messages = local_provider.requests[0].messages

    system_contents = [
        message.content
        for message in local_messages
        if message.role is MessageRole.SYSTEM
    ]

    combined_system_content = "\n".join(system_contents).casefold()

    assert "s7ven is a personal ai assistant." in combined_system_content
    assert "friendly" in combined_system_content

    search_mock.assert_awaited_once()
    assert pipeline.context.message_count == 2
    assert len(api_provider.requests) == 0
