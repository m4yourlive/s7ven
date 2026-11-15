
"""OpenAI-compatible asynchronous provider for S7VEN."""

from collections.abc import AsyncIterator
from typing import Any

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
    AIProviderError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRateLimitError,
    AIResponseError,
)
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import GenerationRequest, GenerationResponse


class OpenAIProvider(AIProvider):
    """Asynchronous provider using the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key cannot be empty.")

        if not model or not model.strip():
            raise ValueError("Model name cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self._model = model.strip()
        self._client = AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=base_url,
            timeout=timeout,
        )
        self._closed = False

    @property
    def name(self) -> str:
        return "openai"

    def _build_messages(
        self,
        request: GenerationRequest,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ]

    def _build_request_options(
        self,
        request: GenerationRequest,
    ) -> dict[str, Any]:
        return {
            "model": request.model or self._model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self._ensure_open()

        try:
            response = await self._client.chat.completions.create(
                **self._build_request_options(request),
            )

            if not response.choices:
                raise AIResponseError(
                    "OpenAI returned an empty response."
                )

            choice = response.choices[0]
            content = choice.message.content

            if content is None:
                raise AIResponseError(
                    "OpenAI returned an empty message."
                )

            usage: dict[str, int] = {}

            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return GenerationResponse(
                content=content,
                model=response.model,
                provider=self.name,
                finish_reason=choice.finish_reason,
                usage=usage,
            )

        except APITimeoutError as exc:
            raise AIProviderTimeoutError(
                "OpenAI request timed out."
            ) from exc

        except AuthenticationError as exc:
            raise AIAuthenticationError(
                "OpenAI authentication failed."
            ) from exc

        except RateLimitError as exc:
            raise AIRateLimitError(
                "OpenAI rate limit exceeded."
            ) from exc

        except APIConnectionError as exc:
            raise AIProviderUnavailableError(
                "Could not connect to OpenAI."
            ) from exc

        except APIStatusError as exc:
            raise AIProviderError(
                f"OpenAI API returned status {exc.status_code}."
            ) from exc

        except APIError as exc:
            raise AIProviderError(
                "An OpenAI API error occurred."
            ) from exc

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        self._ensure_open()

        stream = None

        try:
            stream = await self._client.chat.completions.create(
                **self._build_request_options(request),
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        except APITimeoutError as exc:
            raise AIProviderTimeoutError(
                "OpenAI streaming request timed out."
            ) from exc

        except AuthenticationError as exc:
            raise AIAuthenticationError(
                "OpenAI authentication failed."
            ) from exc

        except RateLimitError as exc:
            raise AIRateLimitError(
                "OpenAI rate limit exceeded."
            ) from exc

        except APIConnectionError as exc:
            raise AIProviderUnavailableError(
                "Could not connect to OpenAI."
            ) from exc

        except APIStatusError as exc:
            raise AIProviderError(
                f"OpenAI API returned status {exc.status_code}."
            ) from exc

        except APIError as exc:
            raise AIProviderError(
                "An OpenAI streaming error occurred."
            ) from exc

        finally:
            if stream is not None:
                await stream.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIProviderUnavailableError(
                "OpenAI provider is closed."
            )

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._client.close()
