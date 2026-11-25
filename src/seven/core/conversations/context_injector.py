
"""Inject retrieved long-term memories into AI context."""

from seven.core.ai.models import AIMessage, MessageRole
from seven.core.conversations.memory_config import MemoryArchitectureConfig
from seven.core.conversations.memory_contracts import MemoryScoringContext
from seven.core.conversations.memory_normalizer import (
    MemoryCandidateNormalizer,
)
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_token_budget import (
    MemoryTokenBudgetManager,
)


class ContextInjector:
    """Build AI context with relevant long-term memories."""

    def __init__(
        self,
        memory_retriever: MemoryRetriever,
        *,
        max_memories: int | None = None,
        config: MemoryArchitectureConfig | None = None,
    ) -> None:
        if not isinstance(memory_retriever, MemoryRetriever):
            raise TypeError(
                "memory_retriever must be a MemoryRetriever instance."
            )

        if config is None:
            config = MemoryArchitectureConfig()

        if not isinstance(config, MemoryArchitectureConfig):
            raise TypeError(
                "config must be a MemoryArchitectureConfig instance."
            )

        if max_memories is None:
            max_memories = config.max_context_memories

        if (
            isinstance(max_memories, bool)
            or not isinstance(max_memories, int)
            or max_memories < 1
        ):
            raise ValueError("max_memories must be a positive integer.")

        self._memory_retriever = memory_retriever
        self._max_memories = max_memories
        self._token_budget_manager = MemoryTokenBudgetManager(
            max_tokens=config.max_context_tokens,
            chars_per_token=config.token_estimation_chars_per_token,
        )

    async def inject(
        self,
        message: str,
        history: tuple[AIMessage, ...] = (),
        *,
        scoring_context: MemoryScoringContext | None = None,
    ) -> tuple[AIMessage, ...]:
        """Inject contextually relevant memories into message history."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message cannot be empty.")

        if not isinstance(history, tuple):
            raise TypeError("history must be a tuple of AIMessage instances.")

        if any(not isinstance(item, AIMessage) for item in history):
            raise TypeError("history must contain only AIMessage instances.")

        if scoring_context is not None and not isinstance(
            scoring_context,
            MemoryScoringContext,
        ):
            raise TypeError(
                "scoring_context must be a MemoryScoringContext or None."
            )

        memories = await self._memory_retriever.retrieve(
            message,
            limit=self._max_memories,
            context=scoring_context,
        )

        if not memories:
            return history

        unique_memories = (
            MemoryCandidateNormalizer.deduplicate_memories(memories)
        )

        if not unique_memories:
            return history

        memory_context = self._token_budget_manager.build_context(
            unique_memories
        )

        if memory_context is None:
            return history

        memory_message = AIMessage(
            role=MessageRole.SYSTEM,
            content=memory_context,
        )

        return (memory_message, *history)
