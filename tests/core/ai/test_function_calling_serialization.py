
"""Tests for function calling request serialization."""

import pytest

from seven.core.ai.exceptions import AIProviderCapabilityError
from seven.core.ai.models import (
    AIMessage,
    FunctionCall,
    FunctionDefinition,
    GenerationRequest,
    MessageRole,
    ProviderCapabilities,
    ToolChoice,
)
from seven.core.ai.providers.capability_handler import CapabilityHandler
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)


@pytest.fixture
def function_definition():
    return FunctionDefinition(
        name="get_weather",
        description="Retrieve weather information.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    )


@pytest.fixture
def generation_request(function_definition):
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="What is the weather in Jakarta?",
            ),
        ),
        model="test-model",
        tools=(function_definition,),
        tool_choice=ToolChoice.AUTO,
    )


class TestFunctionCallingCapabilityHandler:
    def test_builds_function_calling_options(self, generation_request):
        capabilities = ProviderCapabilities(
            supports_function_calling=True,
        )

        options = CapabilityHandler.build_function_calling_options(
            generation_request,
            capabilities,
            "Test Provider",
        )

        assert options["tool_choice"] == "auto"
        assert len(options["tools"]) == 1
        assert options["tools"][0]["type"] == "function"
        assert (
            options["tools"][0]["function"]["name"]
            == "get_weather"
        )

    def test_serializes_required_tool_choice(self, function_definition):
        request = GenerationRequest(
            messages=(
                AIMessage(
                    role=MessageRole.USER,
                    content="Get the weather.",
                ),
            ),
            model="test-model",
            tools=(function_definition,),
            tool_choice=ToolChoice.REQUIRED,
        )

        options = CapabilityHandler.build_function_calling_options(
            request,
            ProviderCapabilities(supports_function_calling=True),
            "Test Provider",
        )

        assert options["tool_choice"] == "required"

    def test_serializes_none_tool_choice(self, function_definition):
        request = GenerationRequest(
            messages=(
                AIMessage(
                    role=MessageRole.USER,
                    content="Get the weather.",
                ),
            ),
            model="test-model",
            tools=(function_definition,),
            tool_choice=ToolChoice.NONE,
        )

        options = CapabilityHandler.build_function_calling_options(
            request,
            ProviderCapabilities(supports_function_calling=True),
            "Test Provider",
        )

        assert options["tool_choice"] == "none"

    def test_returns_empty_options_without_tools(self):
        request = GenerationRequest(
            messages=(
                AIMessage(
                    role=MessageRole.USER,
                    content="Hello.",
                ),
            ),
            model="test-model",
        )

        options = CapabilityHandler.build_function_calling_options(
            request,
            ProviderCapabilities(),
            "Test Provider",
        )

        assert options == {}

    def test_rejects_unsupported_function_calling(
        self,
        generation_request,
    ):
        with pytest.raises(AIProviderCapabilityError):
            CapabilityHandler.build_function_calling_options(
                generation_request,
                ProviderCapabilities(),
                "Test Provider",
            )


class TestOpenAIProviderSerialization:
    def test_openai_enables_function_calling_by_default(self):
        provider = OpenAIProvider(api_key="test-key")

        assert provider.capabilities.supports_function_calling is True

    def test_openai_respects_explicit_capabilities(self):
        provider = OpenAIProvider(
            api_key="test-key",
            capabilities=ProviderCapabilities(),
        )

        assert provider.capabilities.supports_function_calling is False

    def test_builds_function_calling_request_options(
        self,
        generation_request,
    ):
        provider = OpenAIProvider(api_key="test-key")

        options = provider._build_request_options(generation_request)

        assert options["tool_choice"] == "auto"
        assert options["tools"][0]["type"] == "function"
        assert (
            options["tools"][0]["function"]["name"]
            == "get_weather"
        )

    def test_serializes_assistant_tool_calls(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Jakarta"},
        )

        request = GenerationRequest(
            messages=(
                AIMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(call,),
                ),
            ),
            model="test-model",
        )

        provider = OpenAIProvider(api_key="test-key")
        messages = provider._build_messages(request)

        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["tool_calls"][0]["id"] == "call_123"
        assert (
            messages[0]["tool_calls"][0]["function"]["name"]
            == "get_weather"
        )
        assert (
            messages[0]["tool_calls"][0]["function"]["arguments"]
            == '{"city": "Jakarta"}'
        )

    def test_serializes_tool_result_messages(self):
        request = GenerationRequest(
            messages=(
                AIMessage(
                    role=MessageRole.TOOL,
                    content='{"temperature": 30}',
                    tool_call_id="call_123",
                ),
            ),
            model="test-model",
        )

        provider = OpenAIProvider(api_key="test-key")
        messages = provider._build_messages(request)

        assert messages == [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": 30}',
            }
        ]

    def test_rejects_unsupported_function_calling(self, generation_request):
        provider = OpenAIProvider(
            api_key="test-key",
            capabilities=ProviderCapabilities(),
        )

        with pytest.raises(AIProviderCapabilityError):
            provider._build_request_options(generation_request)

    def test_text_only_request_remains_compatible(self):
        request = GenerationRequest(
            messages=(
                AIMessage(
                    role=MessageRole.USER,
                    content="Hello.",
                ),
            ),
            model="test-model",
        )

        provider = OpenAIProvider(api_key="test-key")
        options = provider._build_request_options(request)

        assert options["model"] == "test-model"
        assert options["messages"][0]["content"] == "Hello."
        assert "tools" not in options
        assert "tool_choice" not in options


class TestOpenAICompatibleProviderSerialization:
    def test_function_calling_is_disabled_by_default(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            provider_name="custom-provider",
            base_url="https://example.com/v1",
            model="test-model",
        )

        assert provider.capabilities.supports_function_calling is False

    def test_function_calling_can_be_enabled_explicitly(
        self,
        generation_request,
    ):
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            provider_name="custom-provider",
            base_url="https://example.com/v1",
            model="test-model",
            capabilities=ProviderCapabilities(
                supports_function_calling=True,
            ),
        )

        options = provider._build_request_options(generation_request)

        assert options["tool_choice"] == "auto"
        assert options["tools"][0]["function"]["name"] == "get_weather"
