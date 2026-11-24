
"""Factory for creating local AI model runtimes and providers."""

from __future__ import annotations

from collections.abc import Callable

from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.providers.local_ai_provider import LocalAIProvider
from seven.core.config.config_schema import AIConfig

RuntimeBuilder = Callable[[AIConfig], LocalModelRuntime]


class LocalRuntimeFactory:
    """Create local model runtimes from registered runtime builders."""

    def __init__(self) -> None:
        self._builders: dict[str, RuntimeBuilder] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Runtime name must be a string")

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError("Runtime name must not be empty")

        return normalized

    def register(
        self,
        runtime_name: str,
        builder: RuntimeBuilder,
    ) -> None:
        """Register a builder for a local runtime."""

        name = self._normalize_name(runtime_name)

        if not callable(builder):
            raise TypeError("builder must be callable")

        if name in self._builders:
            raise ValueError(
                f"Runtime builder {name!r} is already registered"
            )

        self._builders[name] = builder

    def unregister(self, runtime_name: str) -> RuntimeBuilder:
        """Remove and return a registered runtime builder."""

        name = self._normalize_name(runtime_name)

        try:
            return self._builders.pop(name)
        except KeyError as exc:
            raise KeyError(
                f"Runtime builder {name!r} is not registered"
            ) from exc

    def has_runtime(self, runtime_name: str) -> bool:
        """Check whether a runtime builder is registered."""

        name = self._normalize_name(runtime_name)
        return name in self._builders

    def list_runtimes(self) -> list[str]:
        """Return registered runtime names."""

        return list(self._builders)

    def create_runtime(self, config: AIConfig) -> LocalModelRuntime:
        """Create a runtime using the configured runtime selection."""

        if not isinstance(config, AIConfig):
            raise TypeError("config must be an AIConfig instance")

        runtime_name = self._normalize_name(config.local_runtime)

        try:
            builder = self._builders[runtime_name]
        except KeyError as exc:
            raise ValueError(
                f"Local runtime {runtime_name!r} is not registered"
            ) from exc

        runtime = builder(config)

        if not isinstance(runtime, LocalModelRuntime):
            raise TypeError(
                "Runtime builder must return a LocalModelRuntime instance"
            )

        actual_name = self._normalize_name(runtime.runtime_name)

        if actual_name != runtime_name:
            raise ValueError(
                f"Runtime builder registered as {runtime_name!r} "
                f"returned runtime {actual_name!r}"
            )

        return runtime

    def create_provider(
        self,
        config: AIConfig,
        provider_name: str = "local",
    ) -> LocalAIProvider:
        """Create a standard AI provider backed by a local runtime."""

        runtime = self.create_runtime(config)

        return LocalAIProvider(
            runtime=runtime,
            provider_name=provider_name,
        )

    def __contains__(self, runtime_name: object) -> bool:
        if not isinstance(runtime_name, str):
            return False

        return runtime_name.strip().lower() in self._builders

    def __len__(self) -> int:
        return len(self._builders)
