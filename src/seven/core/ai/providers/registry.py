
"""Registry for managing AI providers."""

from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import ProviderCapabilities

class ProviderRegistry:
    """Registry for managing available AI providers."""

    def __init__(self) -> None:
        self._providers: dict[str, AIProvider] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize and validate a provider identifier."""
        if not isinstance(name, str):
            raise TypeError("Provider name must be a string.")

        normalized = name.strip().casefold()

        if not normalized:
            raise ValueError("Provider name cannot be empty.")

        return normalized

    def register(self, provider: AIProvider) -> None:
        """Register an AI provider."""
        if not isinstance(provider, AIProvider):
            raise TypeError("provider must implement AIProvider.")

        name = self._normalize_name(provider.name)

        if not isinstance(provider.capabilities, ProviderCapabilities):
            raise TypeError(
                "Provider capabilities must be ProviderCapabilities."
            )

        if name in self._providers:
            raise ValueError(
                f"Provider '{name}' is already registered."
            )

        self._providers[name] = provider

    def get(self, name: str) -> AIProvider:
        """Retrieve a registered AI provider."""
        key = self._normalize_name(name)

        try:
            return self._providers[key]
        except KeyError as exc:
            raise KeyError(
                f"Provider '{key}' is not registered."
            ) from exc

    def get_capabilities(
        self,
        name: str,
    ) -> ProviderCapabilities:
        """Retrieve capabilities for a registered provider."""
        return self.get(name).capabilities

    def unregister(self, name: str) -> AIProvider:
        """Remove and return a registered provider."""
        key = self._normalize_name(name)

        try:
            return self._providers.pop(key)
        except KeyError as exc:
            raise KeyError(
                f"Provider '{key}' is not registered."
            ) from exc

    def list_providers(self) -> tuple[str, ...]:
        """Return registered provider names."""
        return tuple(self._providers)

    def has_provider(self, name: str) -> bool:
        """Check whether a provider is registered."""
        key = self._normalize_name(name)
        return key in self._providers

    def __contains__(self, name: object) -> bool:
        """Support provider membership checks."""
        if not isinstance(name, str):
            return False

        return self.has_provider(name)

    def __len__(self) -> int:
        """Return the number of registered providers."""
        return len(self._providers)