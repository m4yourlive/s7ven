"""AI provider adapter for local model runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from seven.core.ai.exceptions import (
    AIError,
    AIProviderError,
    AIProviderNetworkError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRequestError,
)
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)


class LocalAIProvider(AIProvider):
    """Adapt a local model runtime to the standard AI provider interface."""

    def __init__(
        self,
        runtime: LocalModelRuntime,
        provider_name: str = "local",
    ) -> None:
        if not isinstance(runtime, LocalModelRuntime):
            raise TypeError("runtime must implement LocalModelRuntime")

        normalized_name = provider_name.strip().lower()

        if not normalized_name:
            raise ValueError("provider_name must not be empty")

        self._runtime = runtime
        self._provider_name = normalized_name
        self._closed = False

    @property
    def name(self) -> str:
        """Return the provider identifier."""
        return self._provider_name

    @property
    def runtime(self) -> LocalModelRuntime:
        """Return the underlying local runtime."""
        return self._runtime

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities supported by the runtime."""
        return self._runtime.capabilities

    @property
    def is_closed(self) -> bool:
        """Return whether the provider has been closed."""
        return self._closed

    def _ensure_open(self) -> None:
        """Ensure the local provider is available."""
        if self._closed:
            raise AIProviderUnavailableError(
                f"{self.name} provider is closed."
            )

    @staticmethod
    def _normalize_runtime_error(
        error: Exception,
        *,
        operation: str,
        provider_name: str,
    ) -> None:
        """Translate common runtime failures into S7VEN exceptions."""
        if isinstance(error, AIError):
            raise error

        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            raise AIProviderTimeoutError(
                f"{provider_name} runtime {operation} timed out."
            ) from error

        if isinstance(error, (ConnectionError, OSError)):
            raise AIProviderNetworkError(
                f"{provider_name} runtime network operation failed."
            ) from error

        raise AIProviderError(
            f"{provider_name} runtime {operation} failed."
        ) from error

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a response using the configured local runtime."""
        self._ensure_open()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "Request must be a GenerationRequest instance."
            )

        try:
            response = await self._runtime.generate(request)
        except Exception as exc:
            self._normalize_runtime_error(
                exc,
                operation="generation",
                provider_name=self.name,
            )

        if not isinstance(response, GenerationResponse):
            raise AIProviderError(
                f"{self.name} runtime returned an invalid generation response."
            )

        return response

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream response chunks from the configured local runtime."""
        self._ensure_open()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "Request must be a GenerationRequest instance."
            )

        try:
            async for chunk in self._runtime.stream(request):
                if not isinstance(chunk, str):
                    raise AIProviderError(
                        f"{self.name} runtime returned an invalid stream chunk."
                    )

                yield chunk
        except Exception as exc:
            self._normalize_runtime_error(
                exc,
                operation="streaming",
                provider_name=self.name,
            )

    async def close(self) -> None:
        """Close the underlying runtime and release its resources."""
        if self._closed:
            return

        self._closed = True

        try:
            await self._runtime.close()
        except Exception as exc:
            self._normalize_runtime_error(
                exc,
                operation="shutdown",
                provider_name=self.name,
            )