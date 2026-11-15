"""Integration tests for the AI response pipeline."""

from unittest.mock import AsyncMock

import pytest

from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIRequestError
from seven.core.ai.models import GenerationResponse, MessageRole
from seven.core.ai.pipeline import AIResponsePipeline
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig


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


@pytest.mark.asyncio
async def test_pipeline_rejects_empty_message(pipeline_components):
    pipeline, provider = pipeline_components

    provider.generate = AsyncMock()

    with pytest.raises(AIRequestError):
        await pipeline.generate("   ")

    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_stream_rejects_empty_message(pipeline_components):
    pipeline, provider = pipeline_components

    with pytest.raises(AIRequestError):
        async for _ in pipeline.stream("   "):
            pass
