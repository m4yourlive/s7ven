"""Tests for the AI retry policy."""

import pytest

from seven.core.ai.exceptions import (
    AIAuthenticationError,
    AIConfigurationError,
    AIProviderNetworkError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AIRateLimitError,
    AIRequestError,
)
from seven.core.ai.retry_policy import RetryPolicy


@pytest.fixture
def retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=4.0,
        backoff_multiplier=2.0,
    )


@pytest.mark.parametrize(
    "error",
    [
        AIProviderNetworkError("Network failure."),
        AIProviderTimeoutError("Provider timeout."),
        AIProviderUnavailableError("Provider unavailable."),
        AIRateLimitError("Rate limit exceeded."),
    ],
)
def test_transient_errors_are_retryable(
    retry_policy: RetryPolicy,
    error: Exception,
) -> None:
    assert retry_policy.is_retryable(error)


@pytest.mark.parametrize(
    "error",
    [
        AIAuthenticationError("Authentication failed."),
        AIConfigurationError("Invalid configuration."),
        AIRequestError("Invalid request."),
        ValueError("Unexpected error."),
    ],
)
def test_permanent_errors_are_not_retryable(
    retry_policy: RetryPolicy,
    error: Exception,
) -> None:
    assert not retry_policy.is_retryable(error)


def test_should_retry_before_max_attempts(
    retry_policy: RetryPolicy,
) -> None:
    error = AIProviderNetworkError("Network failure.")

    assert retry_policy.should_retry(error, 1)
    assert retry_policy.should_retry(error, 2)


def test_should_not_retry_after_max_attempts(
    retry_policy: RetryPolicy,
) -> None:
    error = AIProviderNetworkError("Network failure.")

    assert not retry_policy.should_retry(error, 3)


def test_should_not_retry_permanent_error(
    retry_policy: RetryPolicy,
) -> None:
    error = AIAuthenticationError("Authentication failed.")

    assert not retry_policy.should_retry(error, 1)


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (1, 0.5),
        (2, 1.0),
        (3, 2.0),
        (4, 4.0),
        (5, 4.0),
    ],
)
def test_exponential_backoff(
    retry_policy: RetryPolicy,
    attempt: int,
    expected: float,
) -> None:
    error = AIProviderNetworkError("Network failure.")

    assert retry_policy.get_delay(error, attempt) == expected


def test_retry_after_attribute_overrides_backoff(
    retry_policy: RetryPolicy,
) -> None:
    error = AIRateLimitError("Rate limit exceeded.")
    error.retry_after = 2.5

    assert retry_policy.get_delay(error, 1) == 2.5


def test_retry_after_header_overrides_backoff(
    retry_policy: RetryPolicy,
) -> None:
    error = AIRateLimitError("Rate limit exceeded.")
    error.headers = {"Retry-After": "3"}

    assert retry_policy.get_delay(error, 1) == 3.0


def test_retry_after_is_capped_by_max_delay(
    retry_policy: RetryPolicy,
) -> None:
    error = AIRateLimitError("Rate limit exceeded.")
    error.retry_after = 10

    assert retry_policy.get_delay(error, 1) == 4.0


def test_invalid_retry_after_uses_exponential_backoff(
    retry_policy: RetryPolicy,
) -> None:
    error = AIRateLimitError("Rate limit exceeded.")
    error.retry_after = "invalid"

    assert retry_policy.get_delay(error, 2) == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": -1},
        {"initial_delay": -0.1},
        {"max_delay": -1},
        {"backoff_multiplier": 0.5},
        {"initial_delay": 5, "max_delay": 2},
    ],
)
def test_invalid_policy_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RetryPolicy(**kwargs)


def test_attempt_must_be_positive(
    retry_policy: RetryPolicy,
) -> None:
    error = AIProviderNetworkError("Network failure.")

    with pytest.raises(ValueError):
        retry_policy.should_retry(error, 0)

    with pytest.raises(ValueError):
        retry_policy.get_delay(error, 0)


def test_attempt_must_be_integer(
    retry_policy: RetryPolicy,
) -> None:
    error = AIProviderNetworkError("Network failure.")

    with pytest.raises(TypeError):
        retry_policy.should_retry(error, 1.5)

    with pytest.raises(TypeError):
        retry_policy.get_delay(error, 1.5)