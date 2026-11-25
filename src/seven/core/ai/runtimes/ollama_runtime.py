
"""Ollama implementation of the local AI model runtime."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx

from seven.core.ai.exceptions import (
    AIAuthenticationError,
    AIConfigurationError,
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRateLimitError,
    AIRequestError,
    AIResponseError,
)
from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)
from seven.core.ai.providers.capability_handler import CapabilityHandler
from seven.core.config.config_schema import AIConfig


class OllamaRuntime(LocalModelRuntime):
    """Execute local AI generation through the Ollama HTTP API."""

    _CAPABILITIES = ProviderCapabilities(
        supports_streaming=True,
        supports_system_messages=True,
        supports_temperature=True,
        supports_max_tokens=True,
        supports_top_p=True,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
    )

    def __init__(
        self,
        config: AIConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(config, AIConfig):
            raise TypeError("config must be an AIConfig instance")

        base_url = config.local_base_url or "http://localhost:11434"
        parsed_url = urlsplit(base_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError(
                "Ollama base URL must be a valid HTTP or HTTPS URL."
            )

        if not config.local_model.strip():
            raise ValueError("Ollama model name cannot be empty.")

        self._model = config.local_model.strip()
        self._timeout = float(config.local_request_timeout)
        self._closed = False

        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(self._timeout),
        )

    @property
    def runtime_name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._CAPABILITIES

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIProviderUnavailableError(
                "Ollama runtime is closed."
            )

    def _build_messages(
        self,
        request: GenerationRequest,
    ) -> list[dict[str, str]]:
        """Convert and validate messages for the Ollama API."""
        CapabilityHandler.validate_system_messages(
            request.messages,
            self.capabilities,
            "Ollama",
        )

        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ]

    def _build_payload(
        self,
        request: GenerationRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build an Ollama request using supported capabilities."""
        options = CapabilityHandler.build_generation_options(
            request,
            self.capabilities,
            max_tokens_key="num_predict",
            include_unset_max_tokens=False,
        )

        return {
            "model": request.model or self._model,
            "messages": self._build_messages(request),
            "stream": stream,
            "options": options,
        }

    @staticmethod
    def _raise_status_error(status_code: int) -> None:
        """Translate Ollama HTTP status codes into application errors."""
        if status_code in (401, 403):
            raise AIAuthenticationError(
                f"Ollama authentication failed (HTTP {status_code})."
            )

        if status_code == 404:
            raise AIConfigurationError(
                "Ollama endpoint or model was not found. "
                "Check the server URL and installed model."
            )

        if status_code in (408, 504):
            raise AIProviderTimeoutError(
                f"Ollama request timed out (HTTP {status_code})."
            )

        if status_code == 429:
            raise AIRateLimitError(
                "Ollama server rejected the request due to rate limiting."
            )

        if status_code in (400, 422):
            raise AIRequestError(
                f"Ollama rejected the generation request "
                f"(HTTP {status_code})."
            )

        if status_code >= 500:
            raise AIProviderUnavailableError(
                f"Ollama server is unavailable (HTTP {status_code})."
            )

        raise AIProviderError(
            f"Ollama request failed (HTTP {status_code})."
        )

    @staticmethod
    def _raise_http_error(exc: httpx.HTTPStatusError) -> None:
        OllamaRuntime._raise_status_error(
            exc.response.status_code
        )

    @staticmethod
    def _validate_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise AIResponseError(
                "Ollama returned an invalid response payload."
            )

        if payload.get("error"):
            raise AIProviderError(
                f"Ollama generation failed: {payload['error']}"
            )

        return payload

    def _build_response(
        self,
        payload: dict[str, Any],
        request: GenerationRequest,
    ) -> GenerationResponse:
        message = payload.get("message")

        if not isinstance(message, dict):
            raise AIResponseError(
                "Ollama returned a response without a message."
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise AIResponseError(
                "Ollama returned an empty or invalid response."
            )

        response_model = payload.get("model")

        if not isinstance(response_model, str) or not response_model.strip():
            response_model = request.model or self._model

        usage: dict[str, int] = {}

        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")

        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
        ):
            usage["prompt_tokens"] = prompt_tokens

        if (
            isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
        ):
            usage["completion_tokens"] = completion_tokens

        if usage:
            usage["total_tokens"] = (
                usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0)
            )

        finish_reason = payload.get("done_reason")

        if not isinstance(finish_reason, str):
            finish_reason = "stop" if payload.get("done") is True else None

        return GenerationResponse(
            content=content,
            model=response_model,
            provider=self.runtime_name,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _map_transport_error(exc: Exception) -> AIProviderError:
        if isinstance(exc, httpx.TimeoutException):
            return AIProviderTimeoutError(
                "Ollama request timed out."
            )

        if isinstance(exc, httpx.ConnectError):
            return AIProviderUnavailableError(
                "Could not connect to the Ollama server. "
                "Make sure Ollama is running."
            )

        if isinstance(exc, httpx.TransportError):
            return AIProviderUnavailableError(
                "Ollama network communication failed."
            )

        return AIProviderError(
            "An unexpected Ollama communication error occurred."
        )

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a complete response using Ollama."""
        self._ensure_open()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "Request must be a GenerationRequest instance."
            )

        try:
            response = await self._client.post(
                "/api/chat",
                json=self._build_payload(request, stream=False),
            )
            response.raise_for_status()

            try:
                payload = response.json()
            except ValueError as exc:
                raise AIResponseError(
                    "Ollama returned invalid JSON."
                ) from exc

            validated_payload = self._validate_payload(payload)

            return self._build_response(
                validated_payload,
                request,
            )

        except httpx.HTTPStatusError as exc:
            self._raise_http_error(exc)

        except httpx.TransportError as exc:
            raise self._map_transport_error(exc) from exc

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream response chunks from Ollama."""
        self._ensure_open()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "Request must be a GenerationRequest instance."
            )

        CapabilityHandler.validate_streaming(
            self.capabilities,
            "Ollama",
        )

        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=self._build_payload(request, stream=True),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        payload = json.loads(line)
                    except ValueError as exc:
                        raise AIResponseError(
                            "Ollama returned an invalid stream chunk."
                        ) from exc

                    validated_payload = self._validate_payload(payload)
                    message = validated_payload.get("message")

                    if not isinstance(message, dict):
                        if validated_payload.get("done") is True:
                            continue

                        raise AIResponseError(
                            "Ollama returned an invalid stream message."
                        )

                    content = message.get("content")

                    if content is None or content == "":
                        continue

                    if not isinstance(content, str):
                        raise AIResponseError(
                            "Ollama returned an invalid stream chunk."
                        )

                    yield content

        except httpx.HTTPStatusError as exc:
            self._raise_http_error(exc)

        except httpx.TransportError as exc:
            raise self._map_transport_error(exc) from exc

    async def close(self) -> None:
        """Close the HTTP client and release runtime resources."""
        if self._closed:
            return

        await self._client.aclose()
        self._closed = True
