
"""Manage token budgets for long-term memory context."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol


class MemoryContent(Protocol):
    """Protocol for objects containing memory text."""

    content: str


class MemoryTokenBudgetManager:
    """Select and format memories within an estimated token budget."""

    CONTEXT_HEADER = "Relevant long-term memories about the user:\n"
    CONTEXT_FOOTER = (
        "\n\nUse these memories when relevant. "
        "Do not assume information that is not provided."
    )

    def __init__(
        self,
        *,
        max_tokens: int = 1200,
        chars_per_token: int = 4,
    ) -> None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
        ):
            raise TypeError("max_tokens must be an integer.")

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")

        if (
            isinstance(chars_per_token, bool)
            or not isinstance(chars_per_token, int)
        ):
            raise TypeError("chars_per_token must be an integer.")

        if chars_per_token < 1:
            raise ValueError("chars_per_token must be positive.")

        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token

    @property
    def max_tokens(self) -> int:
        """Return the configured token budget."""
        return self._max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token usage using a character-based heuristic."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text:
            return 0

        return math.ceil(
            len(text) / self._chars_per_token
        )

    def render_context(
        self,
        memories: Iterable[MemoryContent],
    ) -> str:
        """Render selected memories into a system message."""
        memory_items = tuple(memories)

        if not memory_items:
            raise ValueError("At least one memory is required.")

        memory_content: list[str] = []

        for memory in memory_items:
            content = getattr(memory, "content", None)

            if not isinstance(content, str):
                raise TypeError(
                    "Every memory must contain string content."
                )

            if not content.strip():
                raise ValueError(
                    "Memory content cannot be empty."
                )

            memory_content.append(f"- {content.strip()}")

        return (
            self.CONTEXT_HEADER
            + "\n".join(memory_content)
            + self.CONTEXT_FOOTER
        )

    def fit_memories(
        self,
        memories: Iterable[MemoryContent],
    ) -> tuple[MemoryContent, ...]:
        """Select memories in priority order without exceeding the budget."""
        if isinstance(memories, (str, bytes)):
            raise TypeError("memories must be an iterable of memory objects.")

        selected: list[MemoryContent] = []

        for memory in memories:
            content = getattr(memory, "content", None)

            if not isinstance(content, str):
                raise TypeError(
                    "Every memory must contain string content."
                )

            if not content.strip():
                continue

            candidate = (*selected, memory)
            rendered_context = self.render_context(candidate)

            if self.estimate_tokens(rendered_context) <= self._max_tokens:
                selected.append(memory)

        return tuple(selected)

    def build_context(
        self,
        memories: Iterable[MemoryContent],
    ) -> str | None:
        """Build a context message containing only memories that fit."""
        selected = self.fit_memories(memories)

        if not selected:
            return None

        return self.render_context(selected)
