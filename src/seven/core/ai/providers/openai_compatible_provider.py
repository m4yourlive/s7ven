
"""Generic provider for OpenAI-compatible API services."""

from urllib.parse import urlsplit

from seven.core.ai.models import ProviderCapabilities
from seven.core.ai.providers.openai_provider import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """Connect to third-party APIs using the OpenAI-compatible protocol."""

    def __init__(
        self,
        api_key: str,
        *,
        provider_name: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible API provider."""
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("Provider name cannot be empty.")

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Base URL cannot be empty.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model name cannot be empty.")

        normalized_url = base_url.strip()
        parsed_url = urlsplit(normalized_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError(
                "Base URL must be a valid HTTP or HTTPS URL."
            )

        normalized_name = provider_name.strip().casefold()

        if normalized_name == "openai":
            raise ValueError(
                "Use OpenAIProvider for the OpenAI service."
            )

        if capabilities is not None and not isinstance(
            capabilities,
            ProviderCapabilities,
        ):
            raise TypeError(
                "Capabilities must be a ProviderCapabilities instance."
            )

        super().__init__(
            api_key=api_key,
            model=model,
            base_url=normalized_url,
            timeout=timeout,
            provider_name=normalized_name,
            capabilities=capabilities,
        )
