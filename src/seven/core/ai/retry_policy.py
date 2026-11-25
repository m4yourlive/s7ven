"""Retry policy for transient AI provider failures."""

from dataclasses import dataclass
from collections.abc import Mapping

from seven.core.ai.exceptions import (
    AIProviderNetworkError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRateLimitError,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Define retry behavior for transient provider failures."""

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Validate retry policy configuration."""
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer.")

        for name in ("initial_delay", "max_delay", "backoff_multiplier"):
            value = getattr(self, name)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise TypeError(f"{name} must be a number.")

            if value < 0:
                raise ValueError(f"{name} cannot be negative.")

        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1.")

        if self.max_delay < self.initial_delay:
            raise ValueError(
                "max_delay must be greater than or equal to initial_delay."
            )

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Return whether an exception represents a transient failure."""
        return isinstance(
            error,
            (
                AIProviderNetworkError,
                AIProviderTimeoutError,
                AIProviderUnavailableError,
                AIRateLimitError,
            ),
        )

    def should_retry(
        self,
        error: Exception,
        attempt: int,
    ) -> bool:
        """Determine whether another attempt is allowed."""
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            raise TypeError("attempt must be an integer.")

        if attempt < 1:
            raise ValueError("attempt must be greater than zero.")

        return (
            attempt < self.max_attempts
            and self.is_retryable(error)
        )

    @staticmethod
    def _get_retry_after(error: Exception) -> float | None:
        """Extract a Retry-After value from an exception."""
        retry_after = getattr(error, "retry_after", None)

        if retry_after is None:
            headers = getattr(error, "headers", None)

            if isinstance(headers, Mapping):
                retry_after = next(
                    (
                        value
                        for key, value in headers.items()
                        if str(key).lower() == "retry-after"
                    ),
                    None,
                )

        if retry_after is None:
            return None

        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            return None

        if delay < 0:
            return None

        return delay

    def get_delay(
        self,
        error: Exception,
        attempt: int,
    ) -> float:
        """Calculate the delay before the next retry attempt."""
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            raise TypeError("attempt must be an integer.")

        if attempt < 1:
            raise ValueError("attempt must be greater than zero.")

        retry_after = self._get_retry_after(error)

        if retry_after is not None:
            return min(retry_after, self.max_delay)

        delay = self.initial_delay * (
            self.backoff_multiplier ** (attempt - 1)
        )

        return min(delay, self.max_delay)