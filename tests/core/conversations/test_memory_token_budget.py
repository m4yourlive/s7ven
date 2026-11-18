
"""Tests for long-term memory token budget management."""

from types import SimpleNamespace

import pytest

from seven.core.conversations.memory_config import (
    MemoryArchitectureConfig,
)
from seven.core.conversations.memory_token_budget import (
    MemoryTokenBudgetManager,
)


def make_memory(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def test_estimate_tokens_uses_configured_character_ratio() -> None:
    manager = MemoryTokenBudgetManager(
        max_tokens=100,
        chars_per_token=4,
    )

    assert manager.estimate_tokens("12345678") == 2
    assert manager.estimate_tokens("12345") == 2
    assert manager.estimate_tokens("") == 0


def test_estimate_tokens_rejects_non_string() -> None:
    manager = MemoryTokenBudgetManager()

    with pytest.raises(TypeError):
        manager.estimate_tokens(123)


def test_fit_memories_preserves_priority_order() -> None:
    manager = MemoryTokenBudgetManager(max_tokens=100)

    memories = (
        make_memory("First memory"),
        make_memory("Second memory"),
        make_memory("Third memory"),
    )

    selected = manager.fit_memories(memories)

    assert selected == memories


def test_fit_memories_skips_oversized_memory() -> None:
    manager = MemoryTokenBudgetManager(
        max_tokens=60,
        chars_per_token=4,
    )

    first = make_memory("Short memory")
    oversized = make_memory("A" * 500)
    third = make_memory("Another short memory")

    selected = manager.fit_memories(
        (first, oversized, third)
    )

    assert selected == (first, third)


def test_fit_memories_accounts_for_prompt_overhead() -> None:
    manager = MemoryTokenBudgetManager(
        max_tokens=5,
        chars_per_token=4,
    )

    selected = manager.fit_memories(
        (make_memory("Short memory"),)
    )

    assert selected == ()


def test_build_context_returns_none_when_nothing_fits() -> None:
    manager = MemoryTokenBudgetManager(
        max_tokens=5,
    )

    result = manager.build_context(
        (make_memory("A memory that is too long"),)
    )

    assert result is None


def test_build_context_includes_header_and_footer() -> None:
    manager = MemoryTokenBudgetManager(max_tokens=200)

    result = manager.build_context(
        (make_memory("The user prefers Python."),)
    )

    assert result is not None
    assert result.startswith(
        "Relevant long-term memories about the user:\n"
    )
    assert "- The user prefers Python." in result
    assert result.endswith(
        "Use these memories when relevant. "
        "Do not assume information that is not provided."
    )

    assert manager.estimate_tokens(result) <= manager.max_tokens


def test_fit_memories_skips_empty_content() -> None:
    manager = MemoryTokenBudgetManager(max_tokens=200)

    selected = manager.fit_memories(
        (
            make_memory(""),
            make_memory("   "),
            make_memory("Valid memory"),
        )
    )

    assert len(selected) == 1
    assert selected[0].content == "Valid memory"


def test_fit_memories_rejects_invalid_memory_content() -> None:
    manager = MemoryTokenBudgetManager(max_tokens=200)

    with pytest.raises(TypeError):
        manager.fit_memories((SimpleNamespace(content=123),))


def test_fit_memories_rejects_string_input() -> None:
    manager = MemoryTokenBudgetManager()

    with pytest.raises(TypeError):
        manager.fit_memories("invalid")


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"max_tokens": 0}, ValueError),
        ({"max_tokens": True}, TypeError),
        ({"max_tokens": 1.5}, TypeError),
        ({"chars_per_token": 0}, ValueError),
        ({"chars_per_token": False}, TypeError),
        ({"chars_per_token": 2.5}, TypeError),
    ],
)
def test_token_budget_manager_validates_configuration(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        MemoryTokenBudgetManager(**kwargs)


def test_memory_config_exposes_token_budget_settings() -> None:
    config = MemoryArchitectureConfig()

    assert config.max_context_tokens == 1200
    assert config.token_estimation_chars_per_token == 4


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"max_context_tokens": 0}, ValueError),
        ({"max_context_tokens": True}, TypeError),
        ({"token_estimation_chars_per_token": 0}, ValueError),
        ({"token_estimation_chars_per_token": 2.5}, TypeError),
    ],
)
def test_memory_config_validates_token_budget_settings(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        MemoryArchitectureConfig(**kwargs)
