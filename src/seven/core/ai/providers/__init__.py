"""AI provider implementations and registration utilities."""

from seven.core.ai.interfaces import AIProvider


class ProviderRegistry:
    """Registry for managing available AI providers."""

    def __init__(self) -> None:
        self._providers: dict[str, AIProvider] = {}

    def register(self, provider: AIProvider) -> None:
        """Register an AI provider."""
        if not isinstance(provider, AIProvider):
            raise TypeError("provider must implement AIProvider.")

        name = provider.name.strip().lower()

        if not name:
            raise ValueError("Provider name cannot be empty.")

        if name in self._providers:
            raise ValueError(
                f"Provider '{name}' is already registered."
            )

        self._providers[name] = provider

    def get(self, name: str) -> AIProvider:
        """Retrieve a registered provider by name."""
        key = name.strip().lower()

        try:
            return self._providers[key]
        except KeyError as exc:
            raise KeyError(
                f"Provider '{key}' is not registered."
            ) from exc

    def unregister(self, name: str) -> AIProvider:
        """Remove and return a registered provider."""
        key = name.strip().lower()

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
        return name.strip().lower() in self._providers
        

__all__ = ["AIProvider", "ProviderRegistry"]
