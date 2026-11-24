
"""Core orchestration service for the S7VEN AI engine."""

import asyncio
from collections.abc import AsyncIterator, Callable

from loguru import logger

from seven.core.ai.exceptions import (
    AIError,
    AIGenerationError,
    AIProviderUnavailableError,
    AIRequestError,
    AIResponseError,
)
from seven.core.ai.interfaces import AIProvider
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
    ProviderCapabilities,
)
from seven.core.ai.providers import ProviderRegistry
from seven.core.ai.retry_policy import RetryPolicy
from seven.core.ai.streaming import AIStreamEvent


class AIEngine:
    """Coordinate AI generation through managed providers."""

    def __init__(
        self,
        provider: AIProvider,
        provider_registry: ProviderRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._logger = logger.bind(module="AIEngine")

        self._validate_provider(provider)

        if provider_registry is not None and not isinstance(
            provider_registry,
            ProviderRegistry,
        ):
            raise TypeError(
                "provider_registry must be a ProviderRegistry instance."
            )

        if retry_policy is not None and not isinstance(
            retry_policy,
            RetryPolicy,
        ):
            raise TypeError(
                "retry_policy must be a RetryPolicy instance."
            )

        self._provider = provider
        self._provider_registry = provider_registry
        self._retry_policy = retry_policy or RetryPolicy()
        self._closed = False

        self._condition = asyncio.Condition()
        self._active_operations = 0
        self._switching = False

        self._managed_providers: dict[int, AIProvider] = {
            id(provider): provider,
        }

        if provider_registry is not None:
            if provider_registry.has_provider(provider.name):
                registered_provider = provider_registry.get(provider.name)

                if registered_provider is not provider:
                    raise ValueError(
                        f"Provider '{provider.name}' is already registered "
                        "with a different instance."
                    )
            else:
                provider_registry.register(provider)

        self._logger.debug(
            "AI engine initialized.",
            provider=provider.name,
            retry_attempts=self._retry_policy.max_attempts,
            fallback_enabled=provider_registry is not None,
        )

    @staticmethod
    def _validate_provider(provider: AIProvider) -> None:
        """Validate a provider before activating it."""
        if not isinstance(provider, AIProvider):
            raise TypeError("provider must implement AIProvider.")

        if not isinstance(provider.capabilities, ProviderCapabilities):
            raise TypeError(
                "Provider capabilities must be ProviderCapabilities."
            )

        if not isinstance(provider.name, str) or not provider.name.strip():
            raise ValueError("Provider name must not be empty.")

    @staticmethod
    def _is_provider_closed(provider: AIProvider) -> bool:
        """Check whether a provider has already been closed."""
        is_closed = getattr(provider, "is_closed", False)

        if isinstance(is_closed, bool) and is_closed:
            return True

        return getattr(provider, "_closed", False) is True

    @staticmethod
    def _normalize_generation_error(
        error: Exception,
        provider_name: str,
        operation: str,
    ) -> AIError:
        """Normalize unexpected generation failures."""
        if isinstance(error, AIError):
            return error

        return AIGenerationError(
            f"{provider_name} {operation} failed unexpectedly."
        )

    def _get_provider_candidates(
        self,
        primary_provider: AIProvider,
    ) -> tuple[AIProvider, ...]:
        """Return the primary provider followed by registered fallbacks."""
        candidates = [primary_provider]

        if self._provider_registry is None:
            return tuple(candidates)

        for name in self._provider_registry.list_providers():
            candidate = self._provider_registry.get(name)

            if any(candidate is existing for existing in candidates):
                continue

            candidates.append(candidate)

        return tuple(candidates)

    def _manage_provider(self, provider: AIProvider) -> None:
        """Track a provider used by this engine for lifecycle management."""
        self._managed_providers[id(provider)] = provider

    def _log_failure(
        self,
        *,
        provider: AIProvider,
        operation: str,
        error: Exception,
        attempt: int,
        fallback: bool,
    ) -> None:
        """Log a provider failure without exposing sensitive data."""
        self._logger.error(
            "AI provider operation failed.",
            provider=provider.name,
            operation=operation,
            error_type=type(error).__name__,
            attempt=attempt,
            fallback=fallback,
        )

    @property
    def provider(self) -> AIProvider:
        """Return the active AI provider."""
        return self._provider

    @property
    def provider_name(self) -> str:
        """Return the active provider identifier."""
        return self._provider.name

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the active provider capabilities."""
        return self._provider.capabilities

    @property
    def provider_registry(self) -> ProviderRegistry | None:
        """Return the configured provider registry."""
        return self._provider_registry

    @property
    def retry_policy(self) -> RetryPolicy:
        """Return the configured retry policy."""
        return self._retry_policy

    @property
    def is_closed(self) -> bool:
        """Return whether the engine has been closed."""
        return self._closed

    @property
    def active_operations(self) -> int:
        """Return the number of currently active provider operations."""
        return self._active_operations

    def _ensure_available(self) -> None:
        """Ensure the AI engine is available."""
        if self._closed:
            raise AIProviderUnavailableError("AI engine is closed.")

    async def _acquire_provider(self) -> AIProvider:
        """Acquire the active provider for an operation."""
        async with self._condition:
            while self._switching:
                await self._condition.wait()

            self._ensure_available()
            self._active_operations += 1

            return self._provider

    async def _release_provider(self) -> None:
        """Release a provider operation."""
        async with self._condition:
            self._active_operations -= 1

            if self._active_operations == 0:
                self._condition.notify_all()

    async def _begin_switch(self) -> None:
        """Wait for other switches and block new operations."""
        async with self._condition:
            while self._switching:
                await self._condition.wait()

            self._ensure_available()
            self._switching = True

            try:
                while self._active_operations > 0:
                    await self._condition.wait()
            except BaseException:
                self._switching = False
                self._condition.notify_all()
                raise

    async def _finish_switch(self) -> None:
        """Allow provider operations to resume."""
        async with self._condition:
            self._switching = False
            self._condition.notify_all()

    async def switch_provider(self, name: str) -> AIProvider:
        """Switch to a registered provider safely."""
        self._ensure_available()

        if self._provider_registry is None:
            raise AIProviderUnavailableError(
                "Provider switching requires a ProviderRegistry."
            )

        target_provider = self._provider_registry.get(name)

        return await self.switch_provider_instance(target_provider)

    async def switch_provider_instance(
        self,
        provider: AIProvider,
        on_switch: Callable[[], None] | None = None,
    ) -> AIProvider:
        """Replace the active provider with a specific provider instance."""
        self._ensure_available()
        self._validate_provider(provider)

        if on_switch is not None and not callable(on_switch):
            raise TypeError("on_switch must be callable.")

        if provider is self._provider:
            if on_switch is not None:
                on_switch()
            return self._provider

        previous_provider = self._provider

        await self._begin_switch()

        registry = self._provider_registry
        previous_registered_provider: AIProvider | None = None
        registry_changed = False

        try:
            if registry is not None:
                if registry.has_provider(provider.name):
                    registered_provider = registry.get(provider.name)

                    if registered_provider is not provider:
                        is_managed = (
                            id(registered_provider)
                            in self._managed_providers
                        )

                        if not is_managed:
                            raise ValueError(
                                f"Provider '{provider.name}' is already "
                                "registered with an unmanaged instance."
                            )

                        previous_registered_provider = (
                            registry.unregister(provider.name)
                        )
                        registry_changed = True

                if not registry.has_provider(provider.name):
                    registry.register(provider)
                    registry_changed = True

            if on_switch is not None:
                on_switch()

            self._provider = provider
            self._manage_provider(provider)

            self._logger.info(
                "AI provider switched successfully.",
                previous_provider=previous_provider.name,
                active_provider=provider.name,
            )

        except BaseException:
            if registry is not None and registry_changed:
                if (
                    registry.has_provider(provider.name)
                    and registry.get(provider.name) is provider
                ):
                    registry.unregister(provider.name)

                if (
                    previous_registered_provider is not None
                    and not registry.has_provider(
                        previous_registered_provider.name
                    )
                ):
                    registry.register(previous_registered_provider)

            self._logger.error(
                "AI provider switch failed.",
                previous_provider=previous_provider.name,
                target_provider=provider.name,
            )
            raise

        finally:
            await self._finish_switch()

        return provider

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a validated response with retries and provider fallback."""
        self._ensure_available()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "request must be a GenerationRequest instance."
            )

        primary_provider = await self._acquire_provider()

        try:
            candidates = self._get_provider_candidates(primary_provider)
            last_error: Exception | None = None

            for provider_index, provider in enumerate(candidates):
                self._manage_provider(provider)
                is_fallback = provider_index > 0

                if is_fallback:
                    self._logger.warning(
                        "Activating fallback AI provider.",
                        primary_provider=primary_provider.name,
                        fallback_provider=provider.name,
                        operation="generation",
                    )

                if self._is_provider_closed(provider):
                    last_error = AIProviderUnavailableError(
                        f"{provider.name} provider is closed."
                    )

                    self._logger.warning(
                        "Skipping unavailable AI provider.",
                        provider=provider.name,
                        operation="generation",
                        reason="provider_closed",
                    )
                    continue

                for attempt in range(
                    1,
                    self._retry_policy.max_attempts + 1,
                ):
                    try:
                        response = await provider.generate(request)

                        if not isinstance(response, GenerationResponse):
                            raise AIResponseError(
                                f"{provider.name} returned an invalid "
                                "generation response."
                            )

                        if attempt > 1 or is_fallback:
                            self._logger.info(
                                "AI generation recovered successfully.",
                                provider=provider.name,
                                operation="generation",
                                attempt=attempt,
                                fallback=is_fallback,
                            )

                        return response

                    except Exception as error:
                        normalized_error = self._normalize_generation_error(
                            error,
                            provider.name,
                            "generation",
                        )

                        if normalized_error is not error:
                            self._log_failure(
                                provider=provider,
                                operation="generation",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            raise normalized_error from error

                        last_error = normalized_error
                        provider_closed = self._is_provider_closed(provider)

                        if provider_closed:
                            self._log_failure(
                                provider=provider,
                                operation="generation",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            break

                        if not self._retry_policy.is_retryable(
                            normalized_error
                        ):
                            self._log_failure(
                                provider=provider,
                                operation="generation",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            raise normalized_error

                        if attempt >= self._retry_policy.max_attempts:
                            self._log_failure(
                                provider=provider,
                                operation="generation",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            break

                        delay = self._retry_policy.get_delay(
                            normalized_error,
                            attempt,
                        )

                        self._logger.warning(
                            "Scheduling AI generation retry.",
                            provider=provider.name,
                            operation="generation",
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            max_attempts=self._retry_policy.max_attempts,
                            delay_seconds=delay,
                            error_type=type(normalized_error).__name__,
                        )

                        await asyncio.sleep(delay)

            if last_error is not None:
                self._logger.error(
                    "All AI generation providers failed.",
                    primary_provider=primary_provider.name,
                    provider_count=len(candidates),
                    error_type=type(last_error).__name__,
                )
                raise last_error

            self._logger.error(
                "No available AI provider could handle the request.",
                primary_provider=primary_provider.name,
            )

            raise AIProviderUnavailableError(
                "No available AI provider could handle the request."
            )

        finally:
            await self._release_provider()

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream validated chunks with retries and provider fallback."""
        self._ensure_available()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "request must be a GenerationRequest instance."
            )

        primary_provider = await self._acquire_provider()

        try:
            candidates = self._get_provider_candidates(primary_provider)
            last_error: Exception | None = None

            for provider_index, provider in enumerate(candidates):
                self._manage_provider(provider)
                is_fallback = provider_index > 0

                if is_fallback:
                    self._logger.warning(
                        "Activating fallback AI provider.",
                        primary_provider=primary_provider.name,
                        fallback_provider=provider.name,
                        operation="streaming",
                    )

                if self._is_provider_closed(provider):
                    last_error = AIProviderUnavailableError(
                        f"{provider.name} provider is closed."
                    )

                    self._logger.warning(
                        "Skipping unavailable AI provider.",
                        provider=provider.name,
                        operation="streaming",
                        reason="provider_closed",
                    )
                    continue

                attempt = 1

                while attempt <= self._retry_policy.max_attempts:
                    emitted_chunk = False

                    try:
                        async for chunk in provider.stream(request):
                            if not isinstance(chunk, str):
                                raise AIResponseError(
                                    f"{provider.name} returned an invalid "
                                    "stream chunk."
                                )

                            if chunk:
                                emitted_chunk = True
                                yield chunk

                        if attempt > 1 or is_fallback:
                            self._logger.info(
                                "AI streaming recovered successfully.",
                                provider=provider.name,
                                operation="streaming",
                                attempt=attempt,
                                fallback=is_fallback,
                            )

                        return

                    except Exception as error:
                        normalized_error = self._normalize_generation_error(
                            error,
                            provider.name,
                            "streaming",
                        )

                        if normalized_error is not error:
                            self._log_failure(
                                provider=provider,
                                operation="streaming",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            raise normalized_error from error

                        last_error = normalized_error
                        provider_closed = self._is_provider_closed(provider)

                        if emitted_chunk:
                            self._log_failure(
                                provider=provider,
                                operation="streaming",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )

                            self._logger.error(
                                "Streaming stopped after partial output; "
                                "retry and fallback are disabled.",
                                provider=provider.name,
                                operation="streaming",
                            )
                            raise normalized_error

                        if provider_closed:
                            self._log_failure(
                                provider=provider,
                                operation="streaming",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            break

                        if not self._retry_policy.is_retryable(
                            normalized_error
                        ):
                            self._log_failure(
                                provider=provider,
                                operation="streaming",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            raise normalized_error

                        if attempt >= self._retry_policy.max_attempts:
                            self._log_failure(
                                provider=provider,
                                operation="streaming",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            break

                        delay = self._retry_policy.get_delay(
                            normalized_error,
                            attempt,
                        )

                        self._logger.warning(
                            "Scheduling AI streaming retry.",
                            provider=provider.name,
                            operation="streaming",
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            max_attempts=self._retry_policy.max_attempts,
                            delay_seconds=delay,
                            error_type=type(normalized_error).__name__,
                        )

                        await asyncio.sleep(delay)
                        attempt += 1

            if last_error is not None:
                self._logger.error(
                    "All AI streaming providers failed.",
                    primary_provider=primary_provider.name,
                    provider_count=len(candidates),
                    error_type=type(last_error).__name__,
                )
                raise last_error

            self._logger.error(
                "No available AI provider could stream the response.",
                primary_provider=primary_provider.name,
            )

            raise AIProviderUnavailableError(
                "No available AI provider could stream the response."
            )

        finally:
            await self._release_provider()

    async def _provider_stream_events(
        self,
        provider: AIProvider,
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        """Stream structured events or adapt legacy text streaming."""
        stream_events = getattr(provider, "stream_events", None)

        if callable(stream_events):
            async for event in stream_events(request):
                yield event
            return

        async for chunk in provider.stream(request):
            if not isinstance(chunk, str):
                raise AIResponseError(
                    f"{provider.name} returned an invalid stream chunk."
                )

            if chunk:
                yield AIStreamEvent(content=chunk)

        yield AIStreamEvent(is_final=True)

    async def stream_events(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[AIStreamEvent]:
        """Stream structured content and function-call events."""
        self._ensure_available()

        if not isinstance(request, GenerationRequest):
            raise AIRequestError(
                "request must be a GenerationRequest instance."
            )

        primary_provider = await self._acquire_provider()

        try:
            candidates = self._get_provider_candidates(primary_provider)
            last_error: Exception | None = None

            for provider_index, provider in enumerate(candidates):
                self._manage_provider(provider)
                is_fallback = provider_index > 0

                if is_fallback:
                    self._logger.warning(
                        "Activating fallback AI provider.",
                        primary_provider=primary_provider.name,
                        fallback_provider=provider.name,
                        operation="stream_events",
                    )

                if self._is_provider_closed(provider):
                    last_error = AIProviderUnavailableError(
                        f"{provider.name} provider is closed."
                    )

                    self._logger.warning(
                        "Skipping unavailable AI provider.",
                        provider=provider.name,
                        operation="stream_events",
                        reason="provider_closed",
                    )
                    continue

                attempt = 1

                while attempt <= self._retry_policy.max_attempts:
                    emitted_output = False

                    try:
                        async for event in self._provider_stream_events(
                            provider,
                            request,
                        ):
                            if not isinstance(event, AIStreamEvent):
                                raise AIResponseError(
                                    f"{provider.name} returned an invalid "
                                    "stream event."
                                )

                            has_output = bool(
                                event.content or event.tool_calls
                            )

                            if has_output:
                                emitted_output = True
                                yield event
                            elif event.is_final:
                                yield event

                        if attempt > 1 or is_fallback:
                            self._logger.info(
                                "AI structured streaming recovered "
                                "successfully.",
                                provider=provider.name,
                                operation="stream_events",
                                attempt=attempt,
                                fallback=is_fallback,
                            )

                        return

                    except Exception as error:
                        normalized_error = self._normalize_generation_error(
                            error,
                            provider.name,
                            "structured streaming",
                        )

                        if normalized_error is not error:
                            self._log_failure(
                                provider=provider,
                                operation="stream_events",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            raise normalized_error from error

                        last_error = normalized_error
                        provider_closed = self._is_provider_closed(provider)

                        if emitted_output:
                            self._log_failure(
                                provider=provider,
                                operation="stream_events",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )

                            self._logger.error(
                                "Structured streaming stopped after partial "
                                "output; retry and fallback are disabled.",
                                provider=provider.name,
                                operation="stream_events",
                            )
                            raise normalized_error

                        if provider_closed:
                            self._log_failure(
                                provider=provider,
                                operation="stream_events",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            break

                        if not self._retry_policy.is_retryable(
                            normalized_error
                        ):
                            self._log_failure(
                                provider=provider,
                                operation="stream_events",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            raise normalized_error

                        if attempt >= self._retry_policy.max_attempts:
                            self._log_failure(
                                provider=provider,
                                operation="stream_events",
                                error=normalized_error,
                                attempt=attempt,
                                fallback=is_fallback,
                            )
                            break

                        delay = self._retry_policy.get_delay(
                            normalized_error,
                            attempt,
                        )

                        self._logger.warning(
                            "Scheduling AI structured streaming retry.",
                            provider=provider.name,
                            operation="stream_events",
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            max_attempts=self._retry_policy.max_attempts,
                            delay_seconds=delay,
                            error_type=type(normalized_error).__name__,
                        )

                        await asyncio.sleep(delay)
                        attempt += 1

            if last_error is not None:
                self._logger.error(
                    "All AI structured streaming providers failed.",
                    primary_provider=primary_provider.name,
                    provider_count=len(candidates),
                    error_type=type(last_error).__name__,
                )
                raise last_error

            self._logger.error(
                "No available AI provider could stream structured events.",
                primary_provider=primary_provider.name,
            )

            raise AIProviderUnavailableError(
                "No available AI provider could stream structured events."
            )

        finally:
            await self._release_provider()

    async def close(self) -> None:
        """Close all providers activated by this engine."""
        async with self._condition:
            while self._switching:
                await self._condition.wait()

            if self._closed:
                return

            self._switching = True

            try:
                while self._active_operations > 0:
                    await self._condition.wait()

                self._closed = True
                managed_providers = tuple(
                    self._managed_providers.values()
                )

            except BaseException:
                self._switching = False
                self._condition.notify_all()
                raise

        self._logger.info(
            "Closing AI engine.",
            managed_provider_count=len(managed_providers),
        )

        errors: list[Exception] = []

        try:
            for provider in managed_providers:
                try:
                    await provider.close()

                    self._logger.debug(
                        "AI provider closed successfully.",
                        provider=provider.name,
                    )

                except Exception as exc:
                    errors.append(exc)

                    self._logger.error(
                        "AI provider shutdown failed.",
                        provider=provider.name,
                        error_type=type(exc).__name__,
                    )

        finally:
            await self._finish_switch()

        if errors:
            self._logger.error(
                "AI engine shutdown completed with errors.",
                error_count=len(errors),
            )
            raise errors[0]

        self._logger.info("AI engine closed successfully.")
