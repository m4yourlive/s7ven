
"""Abstract contract for local AI model runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)


class LocalModelRuntime(ABC):
    """Define the interface for local model execution runtimes."""

    _DEFAULT_CAPABILITIES = ProviderCapabilities()

    @property
    @abstractmethod
    def runtime_name(self) -> str:
        """Return the unique runtime identifier."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the runtime's supported generation capabilities."""

        return self._DEFAULT_CAPABILITIES

    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a complete response using a local model."""

    @abstractmethod
    def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream generated response chunks from a local model."""

    @abstractmethod
    async def close(self) -> None:
        """Release runtime resources."""
