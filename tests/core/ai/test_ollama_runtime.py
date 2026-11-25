
"""Tests for the Ollama local AI runtime."""

import json

import httpx
import pytest

from seven.core.ai.exceptions import (
    AIConfigurationError,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRequestError,
    AIResponseError,
)
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.runtimes import OllamaRuntime
from seven.core.config.config_schema import AIConfig


@pytest.fixture
def config() -> AIConfig:
    return AIConfig(
        mode="local",
        local_runtime="ollama",
        local_model="llama3.2",
        local_base_url="http://localhost:11434",
        local_request_timeout=30,
    )


@pytest.fixture
def generation_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
            ),
            AIMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
        model="llama3.2",
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.1,
    )


def make_runtime(
    config: AIConfig,
    handler,
) -> OllamaRuntime:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=config.local_base_url,
    )

    return OllamaRuntime(config, client=client)


def success_payload() -> dict:
    return {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "Hello from Ollama.",
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


@pytest.mark.asyncio
async def test_runtime_identity_and_capabilities(
    config: AIConfig,
) -> None:
    runtime = OllamaRuntime(config)

    assert runtime.runtime_name == "ollama"
    assert runtime.capabilities.supports_streaming is True
    assert runtime.capabilities.supports_system_messages is True
    assert runtime.capabilities.supports_temperature is True
    assert runtime.capabilities.supports_max_tokens is True
    assert runtime.capabilities.supports_top_p is True
    assert runtime.capabilities.supports_frequency_penalty is False
    assert runtime.capabilities.supports_presence_penalty is False
    assert runtime.is_closed is False

    await runtime.close()


def test_runtime_rejects_invalid_config() -> None:
    with pytest.raises(TypeError, match="AIConfig"):
        OllamaRuntime(None)


def test_runtime_rejects_invalid_base_url() -> None:
    config = AIConfig(
        mode="local",
        local_base_url="invalid-url",
    )

    with pytest.raises(ValueError, match="valid HTTP or HTTPS"):
        OllamaRuntime(config)


def test_build_payload(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    payload = runtime._build_payload(
        generation_request,
        stream=False,
    )

    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Hello.",
        },
    ]
    assert payload["options"] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 128,
    }


def test_build_payload_omits_unsupported_penalties(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    payload = runtime._build_payload(
        generation_request,
        stream=False,
    )

    assert "frequency_penalty" not in payload["options"]
    assert "presence_penalty" not in payload["options"]


def test_build_payload_omits_max_tokens_when_unset(
    config: AIConfig,
) -> None:
    runtime = OllamaRuntime(config)

    generation_request = GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
        model="llama3.2",
    )

    payload = runtime._build_payload(
        generation_request,
        stream=False,
    )

    assert "num_predict" not in payload["options"]


def test_build_response(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    response = runtime._build_response(
        success_payload(),
        generation_request,
    )

    assert response.content == "Hello from Ollama."
    assert response.model == "llama3.2"
    assert response.provider == "ollama"
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_build_response_rejects_missing_message(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    with pytest.raises(AIResponseError, match="without a message"):
        runtime._build_response({}, generation_request)


def test_build_response_rejects_empty_content(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    with pytest.raises(AIResponseError, match="empty or invalid"):
        runtime._build_response(
            {
                "message": {
                    "content": "",
                },
            },
            generation_request,
        )


@pytest.mark.asyncio
async def test_generate(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/api/chat"

        payload = json.loads(http_request.content)

        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False
        assert "frequency_penalty" not in payload["options"]
        assert "presence_penalty" not in payload["options"]

        return httpx.Response(
            200,
            json=success_payload(),
        )

    runtime = make_runtime(config, handler)

    response = await runtime.generate(generation_request)

    assert response.content == "Hello from Ollama."
    assert response.provider == "ollama"

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_rejects_invalid_request(
    config: AIConfig,
) -> None:
    runtime = OllamaRuntime(config)

    with pytest.raises(AIRequestError, match="GenerationRequest"):
        await runtime.generate(None)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_maps_http_404(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    runtime = make_runtime(config, handler)

    with pytest.raises(AIConfigurationError):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_maps_http_400(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    runtime = make_runtime(config, handler)

    with pytest.raises(AIRequestError):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_maps_server_error(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    runtime = make_runtime(config, handler)

    with pytest.raises(AIProviderUnavailableError):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_maps_connection_error(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    runtime = make_runtime(config, handler)

    with pytest.raises(AIProviderUnavailableError):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_maps_timeout(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    runtime = make_runtime(config, handler)

    with pytest.raises(AIProviderTimeoutError):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_rejects_invalid_json(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content="not-json",
        )

    runtime = make_runtime(config, handler)

    with pytest.raises(AIResponseError, match="invalid JSON"):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_generate_rejects_ollama_error_payload(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": "model not found"},
        )

    runtime = make_runtime(config, handler)

    with pytest.raises(AIProviderError, match="model not found"):
        await runtime.generate(generation_request)

    await runtime.close()


@pytest.mark.asyncio
async def test_stream(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    chunks = [
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "Hello ",
            },
            "done": False,
        },
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "from Ollama.",
            },
            "done": False,
        },
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
            },
            "done": True,
            "done_reason": "stop",
        },
    ]

    body = "\n".join(
        json.dumps(chunk)
        for chunk in chunks
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)

        assert payload["stream"] is True
        assert "frequency_penalty" not in payload["options"]
        assert "presence_penalty" not in payload["options"]

        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )

    runtime = make_runtime(config, handler)

    result = [
        chunk
        async for chunk in runtime.stream(generation_request)
    ]

    assert result == ["Hello ", "from Ollama."]

    await runtime.close()


@pytest.mark.asyncio
async def test_stream_rejects_invalid_request(
    config: AIConfig,
) -> None:
    runtime = OllamaRuntime(config)

    with pytest.raises(AIRequestError, match="GenerationRequest"):
        async for _ in runtime.stream(None):
            pass

    await runtime.close()


@pytest.mark.asyncio
async def test_stream_rejects_invalid_json_chunk(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content="invalid-json\n",
            headers={"content-type": "application/x-ndjson"},
        )

    runtime = make_runtime(config, handler)

    with pytest.raises(AIResponseError, match="invalid stream chunk"):
        async for _ in runtime.stream(generation_request):
            pass

    await runtime.close()


@pytest.mark.asyncio
async def test_stream_maps_http_error(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    runtime = make_runtime(config, handler)

    with pytest.raises(AIProviderUnavailableError):
        async for _ in runtime.stream(generation_request):
            pass

    await runtime.close()


@pytest.mark.asyncio
async def test_close_is_idempotent(
    config: AIConfig,
) -> None:
    runtime = OllamaRuntime(config)

    await runtime.close()
    await runtime.close()

    assert runtime.is_closed is True


@pytest.mark.asyncio
async def test_closed_runtime_rejects_generation(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    await runtime.close()

    with pytest.raises(AIProviderUnavailableError, match="closed"):
        await runtime.generate(generation_request)


@pytest.mark.asyncio
async def test_closed_runtime_rejects_stream(
    config: AIConfig,
    generation_request: GenerationRequest,
) -> None:
    runtime = OllamaRuntime(config)

    await runtime.close()

    with pytest.raises(AIProviderUnavailableError, match="closed"):
        async for _ in runtime.stream(generation_request):
            pass
