
"""Tests for AI engine structured logging."""

from unittest.mock import patch

import pytest

from seven.core.ai.exceptions import (
    AIProviderNetworkError,
    AIGenerationError,
)
from seven.core.ai.retry_policy import RetryPolicy


def test_retry_policy_classifies_network_error_as_retryable():
    error = AIProviderNetworkError("Network unavailable.")

    assert RetryPolicy.is_retryable(error) is True


def test_retry_policy_classifies_generation_error_as_non_retryable():
    error = AIGenerationError("Generation failed.")

    assert RetryPolicy.is_retryable(error) is False


def test_retry_policy_calculates_exponential_backoff():
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        backoff_multiplier=2.0,
    )
    error = AIProviderNetworkError("Network unavailable.")

    assert policy.get_delay(error, 1) == 0.5
    assert policy.get_delay(error, 2) == 1.0
    assert policy.get_delay(error, 3) == 2.0


def test_retry_policy_respects_maximum_delay():
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay=2.0,
        max_delay=3.0,
        backoff_multiplier=2.0,
    )
    error = AIProviderNetworkError("Network unavailable.")

    assert policy.get_delay(error, 3) == 3.0


def test_retry_policy_respects_retry_after():
    policy = RetryPolicy(max_delay=5.0)
    error = AIProviderNetworkError("Rate limited.")
    error.retry_after = 3.0

    assert policy.get_delay(error, 1) == 3.0


def test_retry_policy_caps_retry_after():
    policy = RetryPolicy(max_delay=5.0)
    error = AIProviderNetworkError("Rate limited.")
    error.retry_after = 10.0

    assert policy.get_delay(error, 1) == 5.0


def test_retry_policy_rejects_invalid_attempt():
    policy = RetryPolicy()
    error = AIProviderNetworkError("Network unavailable.")

    with pytest.raises(ValueError):
        policy.should_retry(error, 0)


def test_retry_policy_rejects_boolean_attempt():
    policy = RetryPolicy()
    error = AIProviderNetworkError("Network unavailable.")

    with pytest.raises(TypeError):
        policy.should_retry(error, True)
