
"""Integration tests for the AI response pipeline."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seven.core.ai.context_manager import ConversationContext
from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIGenerationError, AIRequestError
from seven.core.ai.models import GenerationResponse, MessageRole
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig
from seven.core.conversations.context_injector import ContextInjector
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_service import MemoryService


@pytest.fixture
def pipeline_components():
    config = AIConfig()
    prompt_manager = PromptManager(DEFAULT_PROMPTS)

    request_factory = GenerationRequestFactory(
        config=config,
        prompt_manager=prompt_manager,
    )

    provider = OpenAIProvider(
        api_key="test-api-key",
    )

    engine = AIEngine(provider)

    pipeline = AIResponsePipeline(
        engine=engine,
        request_factory=request_factory,
    )

    return pipeline, provider


@pytest.fixture
def memory_context_injector():
    """Build a real context injector with a mocked memory repository."""
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
async def test_pipeline_generate(pipeline_components):
    pipeline, provider = pipeline_components

    expected_response = GenerationResponse(
        content="Hello from S7VEN.",
        model="gpt-4o-mini",
        provider="openai",
    )

    provider.generate = AsyncMock(return_value=expected_response)

    response = await pipeline.generate("Hello!")

    assert response == expected_response

    request = provider.generate.call_args.args[0]

    assert request.messages[0].role == MessageRole.SYSTEM
    assert request.messages[0].content
    assert request.messages[1].role == MessageRole.USER
    assert request.messages[1].content == "Hello!"

    assert pipeline.context.message_count == 2
    assert pipeline.context.history[0].role == MessageRole.USER
    assert pipeline.context.history[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_pipeline_preserves_conversation_history(
    pipeline_components,
):
    pipeline, provider = pipeline_components

    first_response = GenerationResponse(
        content="My name is Seven.",
        model="gpt-4o-mini",
        provider="openai",
    )

    second_response = GenerationResponse(
        content="Your name is Seven.",
        model="gpt-4o-mini",
        provider="openai",
    )

    provider.generate = AsyncMock(
        side_effect=[first_response, second_response]
    )

    await pipeline.generate("What is your name?")
    await pipeline.generate("What is your name?")

    second_request = provider.generate.call_args_list[1].args[0]
    messages = second_request.messages

    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "What is your name?"
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[2].content == "My name is Seven."
    assert messages[3].role == MessageRole.USER
    assert messages[3].content == "What is your name?"

    assert pipeline.context.message_count == 4


@pytest.mark.asyncio
async def test_pipeline_stream(pipeline_components):
    pipeline, provider = pipeline_components

    async def mock_stream(request):
        yield "Hello "
        yield "from "
        yield "S7VEN."

    provider.stream = mock_stream

    chunks = [
        chunk
        async for chunk in pipeline.stream("Hello!")
    ]

    assert chunks == ["Hello ", "from ", "S7VEN."]

    history = pipeline.context.history

    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[1].role == MessageRole.ASSISTANT
    assert history[1].content == "Hello from S7VEN."


@pytest.mark.asyncio
async def test_pipeline_stream_preserves_previous_history(
    pipeline_components,
):
    pipeline, provider = pipeline_components

    first_response = GenerationResponse(
        content="Previous response.",
        model="gpt-4o-mini",
        provider="openai",
    )

    provider.generate = AsyncMock(return_value=first_response)

    await pipeline.generate("First message.")

    async def mock_stream(request):
        assert request.messages[-2].content == "Previous response."
        assert request.messages[-1].content == "Second message."
        yield "Second "
        yield "response."

    provider.stream = mock_stream

    chunks = [
        chunk
        async for chunk in pipeline.stream("Second message.")
    ]

    assert chunks == ["Second ", "response."]

    history = pipeline.context.history

    assert len(history) == 4
    assert history[-1].content == "Second response."


@pytest.mark.asyncio
async def test_pipeline_generate_injects_relevant_memories(
    pipeline_components,
    memory_context_injector,
):
    pipeline, provider = pipeline_components
    injector, search_mock = memory_context_injector

    expected_response = GenerationResponse(
        content="S7VEN is your personal AI assistant.",
        model="gpt-4o-mini",
        provider="openai",
    )

    provider.generate = AsyncMock(return_value=expected_response)

    response = await pipeline.generate(
        "Tell me about S7VEN",
        context_injector=injector,
    )

    assert response == expected_response

    request = provider.generate.call_args.args[0]

    memory_messages = [
        message
        for message in request.messages
        if message.role == MessageRole.SYSTEM
        and "S7VEN is a personal AI assistant." in message.content
    ]

    assert len(memory_messages) == 1
    assert request.messages[-1].role == MessageRole.USER
    assert request.messages[-1].content == "Tell me about S7VEN"

    search_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_stream_injects_relevant_memories(
    pipeline_components,
    memory_context_injector,
):
    pipeline, provider = pipeline_components
    injector, search_mock = memory_context_injector

    captured_request = None

    async def mock_stream(request):
        nonlocal captured_request
        captured_request = request

        yield "S7VEN "
        yield "is your AI assistant."

    provider.stream = mock_stream

    chunks = [
        chunk
        async for chunk in pipeline.stream(
            "Tell me about S7VEN",
            context_injector=injector,
        )
    ]

    assert chunks == ["S7VEN ", "is your AI assistant."]
    assert captured_request is not None

    memory_messages = [
        message
        for message in captured_request.messages
        if message.role == MessageRole.SYSTEM
        and "S7VEN is a personal AI assistant." in message.content
    ]

    assert len(memory_messages) == 1
    assert captured_request.messages[-1].role == MessageRole.USER
    assert captured_request.messages[-1].content == "Tell me about S7VEN"

    search_mock.assert_awaited_once()

    assert pipeline.context.message_count == 2


@pytest.mark.asyncio
async def test_pipeline_rejects_empty_message(pipeline_components):
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock()

    with pytest.raises(AIRequestError):
        await pipeline.generate("   ")

    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_stream_rejects_empty_message(
    pipeline_components,
):
    pipeline, provider = pipeline_components

    with pytest.raises(AIRequestError):
        async for _ in pipeline.stream("   "):
            pass


def test_context_limits_conversation_history():
    context = ConversationContext(max_messages=2)

    context.add_user_message("Message one.")
    context.add_assistant_message("Response one.")
    context.add_user_message("Message two.")

    assert context.message_count == 2

    history = context.history

    assert history[0].content == "Response one."
    assert history[1].content == "Message two."


def test_context_preserves_system_prompt_when_cleared():
    context = ConversationContext(
        system_prompt="You are Seven.",
    )

    context.add_user_message("Hello.")
    context.add_assistant_message("Hi.")

    context.clear()

    assert context.message_count == 0
    assert context.system_prompt == "You are Seven."


@pytest.mark.asyncio
async def test_pipeline_stream_does_not_persist_partial_response_on_error(
    pipeline_components,
):
    pipeline, provider = pipeline_components

    async def failing_stream(generation_request):
        yield "Partial "
        raise RuntimeError("Provider stream failed.")

    provider.stream = failing_stream

    stream = pipeline.stream("Hello!")

    assert await anext(stream) == "Partial "

    with pytest.raises(
        AIGenerationError,
        match="streaming failed unexpectedly",
    ):
        await anext(stream)

    assert pipeline.context.message_count == 0
    assert pipeline.context.history == ()


@pytest.mark.asyncio
async def test_pipeline_stream_does_not_persist_when_cancelled(
    pipeline_components,
):
    pipeline, provider = pipeline_components

    async def mock_stream(generation_request):
        yield "First "
        yield "Second "
        yield "Third"

    provider.stream = mock_stream

    stream = pipeline.stream("Hello!")

    first_chunk = await anext(stream)

    assert first_chunk == "First "

    await stream.aclose()

    assert pipeline.context.history == ()
    assert pipeline.context.message_count == 0


@pytest.mark.asyncio
async def test_pipeline_recovers_after_stream_error(
    pipeline_components,
):
    pipeline, provider = pipeline_components

    async def failing_stream(generation_request):
        yield "Partial "
        raise RuntimeError("Temporary provider failure.")

    provider.stream = failing_stream

    stream = pipeline.stream("First message.")

    assert await anext(stream) == "Partial "

    with pytest.raises(AIGenerationError):
        await anext(stream)

    expected_response = GenerationResponse(
        content="Recovered successfully.",
        model="gpt-4o-mini",
        provider="openai",
    )

    provider.generate = AsyncMock(return_value=expected_response)

    response = await pipeline.generate("Second message.")

    assert response == expected_response
    assert pipeline.context.message_count == 2
    assert pipeline.context.history[0].content == "Second message."
    assert (
        pipeline.context.history[1].content
        == "Recovered successfully."
    )
