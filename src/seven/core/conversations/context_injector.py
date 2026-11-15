
"""Inject retrieved long-term memories into AI context."""

from seven.core.ai.models import AIMessage, MessageRole
from seven.core.conversations.memory_retriever import MemoryRetriever


class ContextInjector:
    """Build AI context with relevant long-term memories."""

    def __init__(
        self,
        memory_retriever: MemoryRetriever,
        *,
        max_memories: int = 5,
    ) -> None:
        if not isinstance(memory_retriever, MemoryRetriever):
            raise TypeError(
                "memory_retriever must be a MemoryRetriever instance."
            )

        if (
            not isinstance(max_memories, int)
            or isinstance(max_memories, bool)
            or max_memories < 1
        ):
            raise ValueError("max_memories must be a positive integer.")

        self._memory_retriever = memory_retriever
        self._max_memories = max_memories

    async def inject(
        self,
        message: str,
        history: tuple[AIMessage, ...] = (),
    ) -> tuple[AIMessage, ...]:
        """Inject relevant memories into the message context."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message cannot be empty.")

        if not isinstance(history, tuple):
            raise TypeError("history must be a tuple of AIMessage instances.")

        if any(not isinstance(item, AIMessage) for item in history):
            raise TypeError("history must contain only AIMessage instances.")

        memories = await self._memory_retriever.retrieve(
            message,
            limit=self._max_memories,
        )

        if not memories:
            return history

        memory_content = "\n".join(
            f"- {memory.content}"
            for memory in memories
        )

        memory_message = AIMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Relevant long-term memories about the user:\n"
                f"{memory_content}\n\n"
                "Use these memories when relevant. "
                "Do not assume information that is not provided."
            ),
        )

        return (memory_message, *history)
