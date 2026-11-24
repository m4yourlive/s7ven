
"""Abstract interface for AI language model providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)

class AIProvider(ABC):
    """Contract that every AI provider must implement."""

    _DEFAULT_CAPABILITIES = ProviderCapabilities()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider identifier."""
        raise NotImplementedError

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities supported by this provider."""
        return self._DEFAULT_CAPABILITIES

    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a complete AI response."""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream an AI response in text chunks."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
        raise NotImplementedError