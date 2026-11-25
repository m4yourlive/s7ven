
"""Tests for AI function calling domain models."""

from dataclasses import FrozenInstanceError

import pytest

from seven.core.ai.models import (
    AIMessage,
    FunctionCall,
    FunctionDefinition,
    GenerationRequest,
    GenerationResponse,
    MessageRole,
    ProviderCapabilities,
    ToolChoice,
)


def create_function() -> FunctionDefinition:
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


def create_request(**kwargs) -> GenerationRequest:
    values = {
        "messages": (
            AIMessage(
                role=MessageRole.USER,
                content="What is the weather?",
            ),
        ),
        "model": "test-model",
    }
    values.update(kwargs)
    return GenerationRequest(**values)


class TestFunctionDefinition:
    def test_creates_function_definition(self):
        function = create_function()

        assert function.name == "get_weather"
        assert function.description
        assert function.parameters["type"] == "object"

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            FunctionDefinition(
                name=" ",
                description="Retrieve weather.",
            )

    def test_rejects_empty_description(self):
        with pytest.raises(ValueError):
            FunctionDefinition(
                name="get_weather",
                description=" ",
            )

    def test_rejects_invalid_parameters(self):
        with pytest.raises(TypeError):
            FunctionDefinition(
                name="get_weather",
                description="Retrieve weather.",
                parameters=[],
            )

    def test_parameters_are_immutable(self):
        function = create_function()

        with pytest.raises(TypeError):
            function.parameters["new"] = "value"


class TestFunctionCall:
    def test_creates_function_call(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Jakarta"},
        )

        assert call.call_id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments["city"] == "Jakarta"

    def test_rejects_empty_call_id(self):
        with pytest.raises(ValueError):
            FunctionCall(
                call_id=" ",
                name="get_weather",
            )

    def test_rejects_empty_function_name(self):
        with pytest.raises(ValueError):
            FunctionCall(
                call_id="call_123",
                name=" ",
            )

    def test_rejects_invalid_arguments(self):
        with pytest.raises(TypeError):
            FunctionCall(
                call_id="call_123",
                name="get_weather",
                arguments=[],
            )

    def test_arguments_are_immutable(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Jakarta"},
        )

        with pytest.raises(TypeError):
            call.arguments["city"] = "Bandung"


class TestAIMessageFunctionCalling:
    def test_assistant_message_can_contain_tool_calls(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Jakarta"},
        )

        message = AIMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(call,),
        )

        assert message.tool_calls == (call,)

    def test_tool_message_requires_call_id(self):
        with pytest.raises(ValueError):
            AIMessage(
                role=MessageRole.TOOL,
                content="Sunny",
            )

    def test_tool_message_accepts_call_id(self):
        message = AIMessage(
            role=MessageRole.TOOL,
            content="Sunny",
            tool_call_id="call_123",
        )

        assert message.tool_call_id == "call_123"

    def test_non_assistant_message_cannot_contain_tool_calls(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
        )

        with pytest.raises(ValueError):
            AIMessage(
                role=MessageRole.USER,
                content="Hello",
                tool_calls=(call,),
            )

    def test_tool_message_cannot_contain_tool_calls(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
        )

        with pytest.raises(ValueError):
            AIMessage(
                role=MessageRole.TOOL,
                content="Result",
                tool_call_id="call_123",
                tool_calls=(call,),
            )


class TestGenerationRequestFunctionCalling:
    def test_request_accepts_function_definitions(self):
        function = create_function()

        request = create_request(
            tools=(function,),
        )

        assert request.tools == (function,)
        assert request.tool_choice == ToolChoice.AUTO

    def test_request_accepts_required_tool_choice(self):
        request = create_request(
            tools=(create_function(),),
            tool_choice=ToolChoice.REQUIRED,
        )

        assert request.tool_choice == ToolChoice.REQUIRED

    def test_request_rejects_required_choice_without_tools(self):
        with pytest.raises(ValueError):
            create_request(
                tool_choice=ToolChoice.REQUIRED,
            )

    def test_request_rejects_duplicate_function_names(self):
        function = create_function()

        with pytest.raises(ValueError):
            create_request(
                tools=(function, function),
            )

    def test_request_rejects_invalid_tool_choice(self):
        with pytest.raises(TypeError):
            create_request(tool_choice="auto")

    def test_request_rejects_invalid_tools_type(self):
        with pytest.raises(TypeError):
            create_request(tools=[create_function()])


class TestGenerationResponseFunctionCalling:
    def test_response_accepts_tool_calls_without_text(self):
        call = FunctionCall(
            call_id="call_123",
            name="get_weather",
            arguments={"city": "Jakarta"},
        )

        response = GenerationResponse(
            content="",
            model="test-model",
            provider="test-provider",
            finish_reason="tool_calls",
            tool_calls=(call,),
        )

        assert response.content == ""
        assert response.tool_calls == (call,)

    def test_response_rejects_empty_content_without_tool_calls(self):
        with pytest.raises(ValueError):
            GenerationResponse(
                content=" ",
                model="test-model",
                provider="test-provider",
            )

    def test_response_rejects_invalid_tool_calls(self):
        with pytest.raises(TypeError):
            GenerationResponse(
                content="",
                model="test-model",
                provider="test-provider",
                tool_calls=["invalid"],
            )


class TestProviderCapabilitiesFunctionCalling:
    def test_function_calling_is_disabled_by_default(self):
        capabilities = ProviderCapabilities()

        assert capabilities.supports_function_calling is False

    def test_function_calling_can_be_enabled(self):
        capabilities = ProviderCapabilities(
            supports_function_calling=True,
        )

        assert capabilities.supports_function_calling is True

    def test_rejects_non_boolean_function_calling_capability(self):
        with pytest.raises(TypeError):
            ProviderCapabilities(
                supports_function_calling="yes",
            )
