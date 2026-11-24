
"""OpenAI-compatible asynchronous provider implementation."""

import json
import sys
from collections.abc import AsyncIterator
from typing import Any, NoReturn

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from seven.core.ai.exceptions import (
    AIAuthenticationError,
    AIConfigurationError,
    AIProviderError,
    AIProviderNetworkError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRateLimitError,
    AIRequestError,
    AIResponseError,
)
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    FunctionCall,
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)
from seven.core.ai.providers.capability_handler import CapabilityHandler
from seven.core.ai.streaming import AIStreamEvent


class OpenAIProvider(AIProvider):
    """Asynchronous provider using an OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
        *,
        provider_name: str = "openai",
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key cannot be empty.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model name cannot be empty.")

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("Timeout must be greater than zero.")

        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("Provider name cannot be empty.")

        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("Base URL must be a string or None.")

        if capabilities is not None and not isinstance(
            capabilities,
            ProviderCapabilities,
        ):
            raise TypeError(
                "Capabilities must be a ProviderCapabilities instance."
            )

        self._model = model.strip()
        self._provider_name = provider_name.strip().casefold()

        if capabilities is not None:
            self._capabilities = capabilities
        else:
            self._capabilities = ProviderCapabilities(
                supports_function_calling=(
                    self._provider_name == "openai"
                ),
            )

        self._client = AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=base_url.strip() if base_url else None,
            timeout=float(timeout),
        )
        self._closed = False

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities configured for this provider."""
        return self._capabilities

    @property
    def _provider_label(self) -> str:
        """Return a readable provider name for error messages."""
        if self.name == "openai":
            return "OpenAI"

        return self.name.replace("-", " ").replace("_", " ").title()

    def _build_messages(
        self,
        request: GenerationRequest,
    ) -> list[dict[str, Any]]:
        """Convert messages into OpenAI-compatible format."""
        CapabilityHandler.validate_system_messages(
            request.messages,
            self.capabilities,
            self._provider_label,
        )

        messages: list[dict[str, Any]] = []

        for message in request.messages:
            if message.role.value == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
                continue

            serialized_message: dict[str, Any] = {
                "role": message.role.value,
                "content": message.content or None,
            }

            if message.tool_calls:
                serialized_message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                dict(call.arguments),
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]

            messages.append(serialized_message)

        return messages

    def _build_request_options(
        self,
        request: GenerationRequest,
    ) -> dict[str, Any]:
        """Build request options supported by the active provider."""
        options: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": self._build_messages(request),
        }

        options.update(
            CapabilityHandler.build_generation_options(
                request,
                self.capabilities,
                max_tokens_key="max_tokens",
                include_unset_max_tokens=True,
            )
        )

        options.update(
            CapabilityHandler.build_function_calling_options(
                request,
                self.capabilities,
                self._provider_label,
            )
        )

        return options

    def _parse_tool_calls(
        self,
        response_message: Any,
    ) -> tuple[FunctionCall, ...]:
        """Parse function calls returned by the provider."""
        raw_tool_calls = getattr(response_message, "tool_calls", None)

        if raw_tool_calls is None:
            return ()

        if not isinstance(raw_tool_calls, (list, tuple)):
            raise AIResponseError(
                f"{self._provider_label} returned invalid tool calls."
            )

        parsed_calls: list[FunctionCall] = []
        seen_ids: set[str] = set()

        for raw_call in raw_tool_calls:
            call_id = getattr(raw_call, "id", None)
            call_type = getattr(raw_call, "type", None)
            function = getattr(raw_call, "function", None)

            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or call_id in seen_ids
            ):
                raise AIResponseError(
                    f"{self._provider_label} returned an invalid tool call ID."
                )

            if call_type != "function":
                raise AIResponseError(
                    f"{self._provider_label} returned an unsupported tool call type."
                )

            name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", None)

            if not isinstance(name, str) or not name.strip():
                raise AIResponseError(
                    f"{self._provider_label} returned a tool call "
                    "without a valid function name."
                )

            if not isinstance(raw_arguments, str):
                raise AIResponseError(
                    f"{self._provider_label} returned invalid function arguments."
                )

            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise AIResponseError(
                    f"{self._provider_label} returned malformed "
                    "function arguments."
                ) from exc

            if not isinstance(arguments, dict):
                raise AIResponseError(
                    f"{self._provider_label} function arguments "
                    "must decode to a JSON object."
                )

            try:
                parsed_call = FunctionCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            except (TypeError, ValueError) as exc:
                raise AIResponseError(
                    f"{self._provider_label} returned an invalid function call."
                ) from exc

            seen_ids.add(call_id)
            parsed_calls.append(parsed_call)

        return tuple(parsed_calls)

    def _validate_response(
        self,
        response: Any,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Validate and normalize a provider response."""
        choices = getattr(response, "choices", None)

        if not choices:
            raise AIResponseError(
                f"{self._provider_label} returned an empty response."
            )

        choice = choices[0]
        response_message = getattr(choice, "message", None)

        if response_message is None:
            raise AIResponseError(
                f"{self._provider_label} returned a response without a message."
            )

        content = getattr(response_message, "content", None)

        if content is None:
            content = ""

        if not isinstance(content, str):
            raise AIResponseError(
                f"{self._provider_label} returned invalid message content."
            )

        tool_calls = self._parse_tool_calls(response_message)

        if not content.strip() and not tool_calls:
            raise AIResponseError(
                f"{self._provider_label} returned an empty or invalid message."
            )

        response_model = getattr(response, "model", None)

        if not isinstance(response_model, str) or not response_model.strip():
            response_model = request.model or self._model

        usage: dict[str, int] = {}
        response_usage = getattr(response, "usage", None)

        if response_usage is not None:
            for field_name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = getattr(response_usage, field_name, None)

                if isinstance(value, int) and not isinstance(value, bool):
                    usage[field_name] = value

        return GenerationResponse(
            content=content,
            model=response_model,
            provider=self.name,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            tool_calls=tool_calls,
        )

    def _raise_status_error(
        self,
        status_code: int,
        *,
        operation: str,
    ) -> NoReturn:
        """Translate HTTP status codes into application exceptions."""
        message_prefix = f"{self._provider_label} {operation}"

        if status_code in (401, 403):
            raise AIAuthenticationError(
                f"{message_prefix} authentication failed "
                f"(HTTP {status_code})."
            )

        if status_code == 404:
            raise AIConfigurationError(
                f"{message_prefix} resource was not found "
                f"(HTTP {status_code}). Check the endpoint and model."
            )

        if status_code in (408, 504):
            raise AIProviderTimeoutError(
                f"{message_prefix} timed out (HTTP {status_code})."
            )

        if status_code == 429:
            raise AIRateLimitError(
                f"{message_prefix} rate limit exceeded."
            )

        if status_code in (400, 422):
            raise AIRequestError(
                f"{message_prefix} rejected the request "
                f"(HTTP {status_code})."
            )

        if status_code >= 500:
            raise AIProviderUnavailableError(
                f"{message_prefix} service is unavailable "
                f"(HTTP {status_code})."
            )

        raise AIProviderError(
            f"{message_prefix} failed with HTTP {status_code}."
        )

    def _normalize_sdk_error(
        self,
        error: Exception,
        *,
        operation: str,
    ) -> NoReturn:
        """Translate an OpenAI SDK exception into an S7VEN exception."""
        if isinstance(error, APITimeoutError):
            raise AIProviderTimeoutError(
                f"{self._provider_label} {operation} timed out."
            ) from error

        if isinstance(error, AuthenticationError):
            raise AIAuthenticationError(
                f"{self._provider_label} authentication failed."
            ) from error

        if isinstance(error, RateLimitError):
            raise AIRateLimitError(
                f"{self._provider_label} rate limit exceeded."
            ) from error

        if isinstance(error, APIConnectionError):
            raise AIProviderNetworkError(
                f"Network connection to {self._provider_label} failed."
            ) from error

        if isinstance(error, APIStatusError):
            self._raise_status_error(
                error.status_code,
                operation=operation,
            )

        if isinstance(error, APIError):
            raise AIProviderError(
                f"An {self._provider_label} API error occurred."
            ) from error

        raise AIProviderError(
            f"An unexpected {self._provider_label} provider error occurred."
        ) from error

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a complete response using the API provider."""
        self._ensure_open()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "Request must be a GenerationRequest instance."
            )

        try:
            response = await self._client.chat.completions.create(
                **self._build_request_options(request),
            )
            return self._validate_response(response, request)

        except (
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APIError,
        ) as exc:
            self._normalize_sdk_error(exc, operation="request")

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream text chunks while preserving the existing provider API."""
        events = self.stream_events(request)

        try:
            async for event in events:
                if event.content:
                    yield event.content
        finally:
            await events.aclose()

    async def stream_events(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        """Stream normalized text and completed function-call events."""
        self._ensure_open()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "Request must be a GenerationRequest instance."
            )

        CapabilityHandler.validate_streaming(
            self.capabilities,
            self._provider_label,
        )

        stream = None
        tool_call_fragments: dict[int, dict[str, Any]] = {}

        try:
            stream = await self._client.chat.completions.create(
                **self._build_request_options(request),
                stream=True,
            )

            async for chunk in stream:
                choices = getattr(chunk, "choices", None)

                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)

                if delta is None:
                    continue

                content = getattr(delta, "content", None)

                if content is not None and content != "":
                    if not isinstance(content, str):
                        raise AIResponseError(
                            f"{self._provider_label} returned "
                            "an invalid stream chunk."
                        )

                    yield AIStreamEvent(content=content)

                raw_tool_calls = getattr(delta, "tool_calls", None)

                if raw_tool_calls is None:
                    continue

                if not isinstance(raw_tool_calls, (list, tuple)):
                    raise AIResponseError(
                        f"{self._provider_label} returned "
                        "invalid streaming tool calls."
                    )

                for raw_call in raw_tool_calls:
                    index = getattr(raw_call, "index", None)

                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index < 0
                    ):
                        raise AIResponseError(
                            f"{self._provider_label} returned "
                            "an invalid streaming tool call index."
                        )

                    fragment = tool_call_fragments.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "",
                            "name": "",
                            "arguments": "",
                        },
                    )

                    call_id = getattr(raw_call, "id", None)
                    call_type = getattr(raw_call, "type", None)
                    function = getattr(raw_call, "function", None)

                    if call_id is not None:
                        if not isinstance(call_id, str):
                            raise AIResponseError(
                                f"{self._provider_label} returned "
                                "an invalid streaming tool call ID."
                            )
                        fragment["id"] += call_id

                    if call_type is not None:
                        if not isinstance(call_type, str):
                            raise AIResponseError(
                                f"{self._provider_label} returned "
                                "an invalid streaming tool call type."
                            )
                        fragment["type"] += call_type

                    if function is not None:
                        name = getattr(function, "name", None)
                        arguments = getattr(function, "arguments", None)

                        if name is not None:
                            if not isinstance(name, str):
                                raise AIResponseError(
                                    f"{self._provider_label} returned "
                                    "an invalid function name fragment."
                                )
                            fragment["name"] += name

                        if arguments is not None:
                            if not isinstance(arguments, str):
                                raise AIResponseError(
                                    f"{self._provider_label} returned "
                                    "an invalid function arguments fragment."
                                )
                            fragment["arguments"] += arguments

            parsed_calls: list[FunctionCall] = []
            seen_ids: set[str] = set()

            for index in sorted(tool_call_fragments):
                fragment = tool_call_fragments[index]

                call_id = fragment["id"]
                call_type = fragment["type"]
                name = fragment["name"]
                raw_arguments = fragment["arguments"]

                if not call_id.strip() or call_id in seen_ids:
                    raise AIResponseError(
                        f"{self._provider_label} returned "
                        "an invalid streaming tool call ID."
                    )

                if call_type != "function":
                    raise AIResponseError(
                        f"{self._provider_label} returned "
                        "an unsupported streaming tool call type."
                    )

                if not name.strip():
                    raise AIResponseError(
                        f"{self._provider_label} returned "
                        "a streaming tool call without a function name."
                    )

                if not raw_arguments:
                    raw_arguments = "{}"

                try:
                    arguments = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise AIResponseError(
                        f"{self._provider_label} returned malformed "
                        "streaming function arguments."
                    ) from exc

                if not isinstance(arguments, dict):
                    raise AIResponseError(
                        f"{self._provider_label} streaming function "
                        "arguments must decode to a JSON object."
                    )

                try:
                    parsed_call = FunctionCall(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                    )
                except (TypeError, ValueError) as exc:
                    raise AIResponseError(
                        f"{self._provider_label} returned "
                        "an invalid streaming function call."
                    ) from exc

                seen_ids.add(call_id)
                parsed_calls.append(parsed_call)

            yield AIStreamEvent(
                tool_calls=tuple(parsed_calls),
                is_final=True,
            )

        except (
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APIError,
        ) as exc:
            self._normalize_sdk_error(
                exc,
                operation="streaming request",
            )

        finally:
            if stream is not None:
                active_exception = sys.exc_info()[0] is not None

                try:
                    await stream.close()
                except Exception as exc:
                    if not active_exception:
                        raise AIProviderError(
                            f"{self._provider_label} stream cleanup failed."
                        ) from exc

    def _ensure_open(self) -> None:
        """Ensure the provider client has not been closed."""
        if self._closed:
            raise AIProviderUnavailableError(
                f"{self._provider_label} provider is closed."
            )

    async def close(self) -> None:
        """Close the provider client."""
        if self._closed:
            return

        self._closed = True
        await self._client.close()
