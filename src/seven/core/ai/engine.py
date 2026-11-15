"""Core orchestration service for the S7VEN AI engine."""

from collections.abc import AsyncIterator

from seven.core.ai.exceptions import (
    AIProviderUnavailableError,
    AIRequestError,
)
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
)


class AIEngine:
    """Coordinate AI generation through a configured provider."""

    def __init__(self, provider: AIProvider) -> None:
        if not isinstance(provider, AIProvider):
            raise TypeError("provider must implement AIProvider.")

        self._provider = provider
        self._closed = False

    @property
    def provider(self) -> AIProvider:
        """Return the active AI provider."""
        return self._provider

    @property
    def is_closed(self) -> bool:
        """Return whether the engine has been closed."""
        return self._closed

    def _ensure_available(self) -> None:
        if self._closed:
            raise AIProviderUnavailableError(
                "AI engine is closed."
            )

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a complete response using the active provider."""
        self._ensure_available()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "request must be a GenerationRequest instance."
            )

        return await self._provider.generate(request)

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream response chunks from the active provider."""
        self._ensure_available()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "request must be a GenerationRequest instance."
            )

        async for chunk in self._provider.stream(request):
            if chunk:
                yield chunk

    async def close(self) -> None:
        """Close the active provider and release its resources."""
        if self._closed:
            return

        self._closed = True
        await self._provider.close()
