
"""Tests for function calling response parsing."""

from types import SimpleNamespace

import pytest

from seven.core.ai.exceptions import AIResponseError
from seven.core.ai.models import (
    AIMessage,
    FunctionCall,
    GenerationRequest,
    MessageRole,
)
from seven.core.ai.providers.openai_provider import OpenAIProvider


def create_request():
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="What is the weather?",
            ),
        ),
        model="test-model",
    )


def create_tool_call(
    *,
    call_id="call_123",
    name="get_weather",
    arguments='{"city": "Jakarta"}',
    call_type="function",
):
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def create_response(
    *,
    content=None,
    tool_calls=None,
    finish_reason="tool_calls",
):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )

    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )

    return SimpleNamespace(
        choices=[choice],
        model="test-model",
        usage=None,
    )


@pytest.fixture
def provider():
    return OpenAIProvider(api_key="test-key")


class TestFunctionCallParsing:
    def test_parses_single_function_call(self, provider):
        response = create_response(
            tool_calls=[create_tool_call()],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].call_id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {
            "city": "Jakarta",
        }

    def test_preserves_provider_call_ids(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(call_id="call_first"),
                create_tool_call(
                    call_id="call_second",
                    name="get_time",
                    arguments='{"timezone": "UTC"}',
                ),
            ],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert [
            call.call_id for call in result.tool_calls
        ] == ["call_first", "call_second"]

    def test_parses_multiple_function_calls(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(),
                create_tool_call(
                    call_id="call_456",
                    name="get_time",
                    arguments='{"timezone": "UTC"}',
                ),
            ],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert len(result.tool_calls) == 2

    def test_parses_function_call_with_empty_arguments(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(arguments="{}"),
            ],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert result.tool_calls[0].arguments == {}

    def test_parses_function_call_with_unicode_arguments(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(
                    arguments='{"city": "Yogyakarta"}',
                ),
            ],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert result.tool_calls[0].arguments["city"] == "Yogyakarta"

    def test_rejects_invalid_tool_call_collection(self, provider):
        response = create_response(tool_calls="invalid")

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_empty_call_id(self, provider):
        response = create_response(
            tool_calls=[create_tool_call(call_id=" ")],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_duplicate_call_ids(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(),
                create_tool_call(),
            ],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_unsupported_tool_call_type(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(call_type="custom"),
            ],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_empty_function_name(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(name=" "),
            ],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_malformed_json_arguments(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(arguments='{"city":'),
            ],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_non_object_arguments(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(arguments='["Jakarta"]'),
            ],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_non_string_arguments(self, provider):
        response = create_response(
            tool_calls=[
                create_tool_call(arguments={"city": "Jakarta"}),
            ],
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())


class TestFunctionCallingResponseValidation:
    def test_accepts_tool_call_without_text(self, provider):
        response = create_response(
            content=None,
            tool_calls=[create_tool_call()],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert result.content == ""
        assert len(result.tool_calls) == 1

    def test_accepts_text_with_tool_calls(self, provider):
        response = create_response(
            content="I will check that.",
            tool_calls=[create_tool_call()],
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert result.content == "I will check that."
        assert len(result.tool_calls) == 1

    def test_accepts_text_only_response(self, provider):
        response = create_response(
            content="The weather is sunny.",
            tool_calls=None,
            finish_reason="stop",
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert result.content == "The weather is sunny."
        assert result.tool_calls == ()

    def test_rejects_empty_response_without_tool_calls(self, provider):
        response = create_response(
            content=None,
            tool_calls=None,
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_missing_choices(self, provider):
        response = SimpleNamespace(
            choices=[],
            model="test-model",
            usage=None,
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_rejects_missing_message(self, provider):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=None,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=None,
        )

        with pytest.raises(AIResponseError):
            provider._validate_response(response, create_request())

    def test_preserves_finish_reason(self, provider):
        response = create_response(
            tool_calls=[create_tool_call()],
            finish_reason="tool_calls",
        )

        result = provider._validate_response(
            response,
            create_request(),
        )

        assert result.finish_reason == "tool_calls"
