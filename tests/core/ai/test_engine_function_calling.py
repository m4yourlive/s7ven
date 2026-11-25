
from collections.abc import AsyncIterator

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import (
    AIProviderUnavailableError,
    AIRequestError,
    AIResponseError,
)
from seven.core.ai.models import (
    AIMessage,
    FunctionCall,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.streaming import AIStreamEvent


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-api-key")


@pytest.fixture
def engine(provider: OpenAIProvider) -> AIEngine:
    return AIEngine(provider)


@pytest.fixture
def generation_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Search for the latest Bitcoin price.",
            ),
        ),
        model="gpt-4o-mini",
    )


@pytest.mark.asyncio
async def test_stream_events_forwards_text_events(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    expected = [
        AIStreamEvent(content="Bitcoin "),
        AIStreamEvent(content="price information."),
        AIStreamEvent(is_final=True),
    ]

    async def mock_stream_events(
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        assert request is generation_request

        for event in expected:
            yield event

    provider.stream_events = mock_stream_events

    events = [
        event
        async for event in engine.stream_events(generation_request)
    ]

    assert events == expected
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_stream_events_forwards_function_calls(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    function_call = FunctionCall(
        call_id="call_weather_123",
        name="get_weather",
        arguments={
            "location": "Jakarta",
        },
    )

    expected = AIStreamEvent(
        tool_calls=(function_call,),
        is_final=True,
    )

    async def mock_stream_events(
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        yield expected

    provider.stream_events = mock_stream_events

    events = [
        event
        async for event in engine.stream_events(generation_request)
    ]

    assert events == [expected]
    assert events[0].tool_calls == (function_call,)


@pytest.mark.asyncio
async def test_stream_events_preserves_multiple_function_calls(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    calls = (
        FunctionCall(
            call_id="call_1",
            name="get_weather",
            arguments={"location": "Jakarta"},
        ),
        FunctionCall(
            call_id="call_2",
            name="get_time",
            arguments={"timezone": "Asia/Jakarta"},
        ),
    )

    expected = AIStreamEvent(
        tool_calls=calls,
        is_final=True,
    )

    async def mock_stream_events(
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        yield expected

    provider.stream_events = mock_stream_events

    events = [
        event
        async for event in engine.stream_events(generation_request)
    ]

    assert len(events) == 1
    assert events[0].tool_calls == calls


@pytest.mark.asyncio
async def test_stream_events_adapts_legacy_provider(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    async def mock_stream(
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Hello "
        yield ""
        yield "Seven."

    provider.stream_events = None

    provider.stream = mock_stream

    events = [
        event
        async for event in engine.stream_events(generation_request)
    ]

    assert events == [
        AIStreamEvent(content="Hello "),
        AIStreamEvent(content="Seven."),
        AIStreamEvent(is_final=True),
    ]


@pytest.mark.asyncio
async def test_stream_events_rejects_invalid_request(
    engine: AIEngine,
) -> None:
    with pytest.raises(AIRequestError):
        async for _ in engine.stream_events("invalid request"):
            pass


@pytest.mark.asyncio
async def test_stream_events_rejects_closed_engine(
    engine: AIEngine,
    generation_request: GenerationRequest,
) -> None:
    await engine.close()

    with pytest.raises(AIProviderUnavailableError):
        async for _ in engine.stream_events(generation_request):
            pass


@pytest.mark.asyncio
async def test_stream_events_rejects_invalid_provider_event(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    async def mock_stream_events(
        request: GenerationRequest,
    ) -> AsyncIterator[object]:
        yield "invalid event"

    provider.stream_events = mock_stream_events

    with pytest.raises(AIResponseError):
        async for _ in engine.stream_events(generation_request):
            pass

    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_stream_events_releases_operation_after_completion(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    async def mock_stream_events(
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        yield AIStreamEvent(content="Completed.")

    provider.stream_events = mock_stream_events

    assert engine.active_operations == 0

    events = [
        event
        async for event in engine.stream_events(generation_request)
    ]

    assert events == [
        AIStreamEvent(content="Completed."),
    ]
    assert engine.active_operations == 0


@pytest.mark.asyncio
async def test_stream_events_releases_operation_after_cancellation(
    engine: AIEngine,
    provider: OpenAIProvider,
    generation_request: GenerationRequest,
) -> None:
    async def mock_stream_events(
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        yield AIStreamEvent(content="First chunk.")

        await asyncio.Event().wait()

    import asyncio

    provider.stream_events = mock_stream_events

    async def consume() -> None:
        async for _ in engine.stream_events(generation_request):
            pass

    task = asyncio.create_task(consume())

    while engine.active_operations == 0:
        await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.active_operations == 0
