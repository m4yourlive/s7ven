
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.ai.exceptions import AIResponseError
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.streaming import AIStreamEvent


@pytest.fixture
def provider():
    return OpenAIProvider(
        api_key="test-api-key",
        model="gpt-4o-mini",
    )


@pytest.fixture
def generation_request():
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Find the weather.",
            ),
        ),
        model="gpt-4o-mini",
    )


def make_chunk(*, content=None, tool_calls=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=delta),
        ],
    )


def make_tool_delta(
    index,
    *,
    call_id=None,
    call_type=None,
    name=None,
    arguments=None,
):
    function = SimpleNamespace(
        name=name,
        arguments=arguments,
    )
    return SimpleNamespace(
        index=index,
        id=call_id,
        type=call_type,
        function=function,
    )


def make_stream(chunks):
    stream = MagicMock()

    async def iterator():
        for chunk in chunks:
            yield chunk

    stream.__aiter__.side_effect = iterator
    stream.close = AsyncMock()
    return stream


@pytest.mark.asyncio
async def test_stream_events_returns_text_events(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(content="Hello "),
        make_chunk(content="Seven!"),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    events = [
        event
        async for event in provider.stream_events(generation_request)
    ]

    assert events == [
        AIStreamEvent(content="Hello "),
        AIStreamEvent(content="Seven!"),
        AIStreamEvent(is_final=True),
    ]
    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_events_reassembles_function_call(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(
            tool_calls=[
                make_tool_delta(
                    0,
                    call_id="call_123",
                    call_type="function",
                    name="get_weather",
                    arguments='{"city":',
                ),
            ],
        ),
        make_chunk(
            tool_calls=[
                make_tool_delta(
                    0,
                    arguments='"Jakarta"}',
                ),
            ],
        ),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    events = [
        event
        async for event in provider.stream_events(generation_request)
    ]

    assert len(events) == 1
    assert events[0].is_final is True
    assert len(events[0].tool_calls) == 1

    call = events[0].tool_calls[0]
    assert call.call_id == "call_123"
    assert call.name == "get_weather"
    assert dict(call.arguments) == {"city": "Jakarta"}

    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_events_reassembles_multiple_function_calls(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(
            tool_calls=[
                make_tool_delta(
                    0,
                    call_id="call_1",
                    call_type="function",
                    name="get_weather",
                    arguments='{"city":"Jakarta"}',
                ),
                make_tool_delta(
                    1,
                    call_id="call_2",
                    call_type="function",
                    name="get_time",
                    arguments='{"timezone":"Asia/Jakarta"}',
                ),
            ],
        ),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    events = [
        event
        async for event in provider.stream_events(generation_request)
    ]

    calls = events[-1].tool_calls

    assert len(calls) == 2
    assert calls[0].call_id == "call_1"
    assert calls[0].name == "get_weather"
    assert calls[1].call_id == "call_2"
    assert calls[1].name == "get_time"

    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_preserves_text_only_api(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(content="Hello "),
        make_chunk(content="Seven!"),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    chunks = [
        chunk
        async for chunk in provider.stream(generation_request)
    ]

    assert chunks == ["Hello ", "Seven!"]
    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_events_rejects_invalid_tool_call_index(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(
            tool_calls=[
                make_tool_delta(
                    -1,
                    call_id="call_123",
                    call_type="function",
                    name="get_weather",
                    arguments="{}",
                ),
            ],
        ),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    with pytest.raises(
        AIResponseError,
        match="invalid streaming tool call index",
    ):
        async for _ in provider.stream_events(generation_request):
            pass

    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_events_rejects_malformed_arguments(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(
            tool_calls=[
                make_tool_delta(
                    0,
                    call_id="call_123",
                    call_type="function",
                    name="get_weather",
                    arguments="{invalid-json",
                ),
            ],
        ),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    with pytest.raises(
        AIResponseError,
        match="malformed streaming function arguments",
    ):
        async for _ in provider.stream_events(generation_request):
            pass

    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_events_closes_after_iteration_error(
    provider,
    generation_request,
):
    mock_stream = MagicMock()
    mock_stream.close = AsyncMock()

    async def failing_iterator():
        yield make_chunk(content="Hello")
        raise RuntimeError("Stream connection interrupted.")

    mock_stream.__aiter__.side_effect = failing_iterator

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    with pytest.raises(
        RuntimeError,
        match="Stream connection interrupted.",
    ):
        async for _ in provider.stream_events(generation_request):
            pass

    mock_stream.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_closes_when_cancelled(
    provider,
    generation_request,
):
    mock_stream = make_stream([
        make_chunk(content="Hello"),
        make_chunk(content="Seven"),
    ])

    provider._client.chat.completions.create = AsyncMock(
        return_value=mock_stream,
    )

    stream = provider.stream(generation_request)

    first_chunk = await anext(stream)
    assert first_chunk == "Hello"

    await stream.aclose()

    mock_stream.close.assert_awaited_once()
